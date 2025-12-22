from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
import csv
from six.moves import cPickle
import traceback
from collections import defaultdict

import utils.LSTM.utils.opts as opts
import models.LSTM as models
from utils.LSTM.dataloader import *
import skimage.io
import utils.LSTM.utils.eval_utils as eval_utils
import utils.LSTM.utils.misc as utils
from utils.LSTM.utils.rewards import init_scorer, get_self_critical_reward
from models.LSTM.modules.loss_wrapper import LossWrapper


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos(if there is) and check if models are compatible
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)
    # CSV logger (repo root)
    csv_log_path = os.path.join('result', 'training_history_data.csv')
    csv_header_written = os.path.exists(csv_log_path)
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = opt.patience

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    device = torch.device('cuda' if torch.cuda.is_available() and not getattr(opt, 'force_cpu', False) else 'cpu')
    if device.type == 'cpu':
        print('using CPU for training (set --force_cpu 0 and install CUDA build to enable GPU)')
    model = models.setup(opt).to(device)
    del opt.vocab
    # Load pretrained weights:
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    
    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    # Wrap with dataparallel
    if device.type == 'cuda':
        dp_model = torch.nn.DataParallel(model)
    else:
        dp_model = model
    dp_model.vocab = getattr(model, 'vocab', None)  # nasty
    if device.type == 'cuda':
        dp_lw_model = torch.nn.DataParallel(lw_model)
    else:
        dp_lw_model = lw_model

    ##########################
    #  Build optimizer
    ##########################
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True
    epoch_start_time = time.time()
    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    # Assure in training mode
    dp_lw_model.train()

    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                
                # If start structure loss training
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False
                if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
                    drop_worst_flag = True
                else:
                    drop_worst_flag = False

                epoch_done = False
                    
            start = time.time()
            if opt.use_warmup and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.to(device) for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag, drop_worst_flag)

            if not drop_worst_flag:
                loss = model_out['loss'].mean()
            else:
                loss = model_out['loss']
                loss = torch.topk(loss, k=int(loss.shape[0] * (1-opt.drop_worst_rate)), largest=False)[0].mean()

            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            train_loss = loss.item()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            if struc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
            elif not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            iteration += 1
            epoch_loss_sum += train_loss
            epoch_loss_count += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('scheduled_sampling_prob', getattr(model, 'ss_prob', 0), iteration)
                if sc_flag:
                    tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                elif struc_flag:
                    tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()
            
            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or \
                (epoch_done and opt.save_every_epoch):
                # eval model
                eval_kwargs = {'split': 'val',
                                'dataset': opt.input_json,
                                'device': device,
                                'language_eval_json': getattr(opt, 'language_eval_json', '')}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1 and lang_stats is not None:
                    current_score = lang_stats.get('CIDEr', lang_stats.get('Bleu_4', -val_loss))
                else:
                    current_score = - val_loss
                val_cider = lang_stats.get('CIDEr', '') if lang_stats is not None else ''
                if lang_stats is not None:
                    print("Validation Results:")
                    for k, v in lang_stats.items():
                        print(f"  {k}: {v:.4f}")

                # Log per-epoch metrics to CSV when an epoch just finished
                if epoch_done:
                    epoch_time = time.time() - epoch_start_time
                    avg_train_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                    if not csv_header_written:
                        with open(csv_log_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['epoch', 'epoch_time_sec', 'train_loss', 'val_loss', 'val_cider'])
                        csv_header_written = True
                    with open(csv_log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, epoch_time, avg_train_loss, val_loss, val_cider])
                    print(f"Saved CSV log to {csv_log_path}")
                    # reset epoch accumulators
                    epoch_start_time = time.time()
                    epoch_loss_sum = 0.0
                    epoch_loss_count = 0

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                # Early stopping on CIDEr (best_flag indicates improvement)
                if best_flag:
                    patience_counter = 0
                    print(f"[EarlyStop] Improved score (CIDEr) to {current_score:.4f}. Reset patience.")
                else:
                    patience_counter += 1
                    print(f"[EarlyStop] No score improvement ({patience_counter}/{early_stop_patience}).")
                    if patience_counter >= early_stop_patience:
                        print("[EarlyStop] Patience exceeded. Stopping training.")
                        utils.save_checkpoint(opt, model, infos, optimizer, histories)
                        break

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score

                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer,
                        append=str(epoch) if opt.save_every_epoch else str(iteration))

                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')


    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


def main():
    opt = opts.parse_opt()
    train(opt)


if __name__ == "__main__":
    if os.name == "nt":
        torch.multiprocessing.set_start_method("spawn", force=True)
    main()
