import matplotlib.pyplot as plt
import csv
import os
import argparse


def read_csv(file_path):
    """Đọc file CSV và trả về các list dữ liệu"""
    epochs = []
    losses = []  # Hoặc Reward đối với GAN
    ciders = []

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))

            # Lấy Loss hoặc Reward tùy tên cột
            if 'Train_Loss' in row:
                losses.append(float(row['Train_Loss']))
            elif 'Avg_Reward' in row:
                losses.append(float(row['Avg_Reward']))

            ciders.append(float(row['CIDEr']))

    return epochs, losses, ciders


def plot_graph(epochs, values, ciders, title, value_label, filename):
    """Vẽ biểu đồ 2 trục: Trái là Loss/Reward, Phải là CIDEr"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Trục 1 (Trái): Loss hoặc Reward
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(value_label, color=color)
    ax1.plot(epochs, values, color=color, marker='o', label=value_label)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Trục 2 (Phải): CIDEr
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation CIDEr', color=color)
    ax2.plot(epochs, ciders, color=color, marker='s', linestyle='--', label='Val CIDEr')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(filename)
    print(f">> Đã lưu biểu đồ: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    # --- 1. VẼ BIỂU ĐỒ MLE ---
    mle_log = os.path.join(args.checkpoint_dir, 'log_mle.csv')
    if os.path.exists(mle_log):
        try:
            epochs, losses, ciders = read_csv(mle_log)
            plot_graph(epochs, losses, ciders,
                       title='MLE Training: Loss vs Val CIDEr',
                       value_label='Train Loss (Lower is better)',
                       filename=os.path.join(args.checkpoint_dir, 'chart_mle.png'))
        except Exception as e:
            print(f"Không thể vẽ biểu đồ MLE: {e}")
    else:
        print(f"Không tìm thấy file {mle_log}")

    # --- 2. VẼ BIỂU ĐỒ GAN ---
    gan_log = os.path.join(args.checkpoint_dir, 'log_gan.csv')
    if os.path.exists(gan_log):
        try:
            epochs, rewards, ciders = read_csv(gan_log)
            plot_graph(epochs, rewards, ciders,
                       title='GAN Training: Reward vs Val CIDEr',
                       value_label='Avg Reward (Higher is better)',
                       filename=os.path.join(args.checkpoint_dir, 'chart_gan.png'))
        except Exception as e:
            print(f"Không thể vẽ biểu đồ GAN: {e}")
    else:
        print(f"Không tìm thấy file {gan_log}")


if __name__ == '__main__':
    main()