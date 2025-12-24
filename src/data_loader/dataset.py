import torch
from torch.utils.data import Dataset
import numpy as np
import os


class ImageDataset(Dataset):
    """
    Class cơ bản chỉ load Feature Ảnh (Dùng cho giai đoạn Inference/Test
    khi không có caption hoặc chỉ cần sinh ảnh).
    """

    def __init__(self, split, args):
        """
        split: 'train', 'val', hoặc 'test'
        args: chứa các đường dẫn config
        """
        self.split = split
        self.max_att_size = 196  # (14x14 regions từ ResNet)

        self.input_fc_dir = 'data/features/fc'
        self.input_att_dir = 'data/features/att'

        # Load list ID image
        id_path = f'data/processed/{split}_images.npy'

        if not os.path.exists(id_path):
            raise FileNotFoundError(f"Không tìm thấy file ID ảnh tại: {id_path}")

        self.image_ids = np.load(id_path)
        print(f"Dataset {split}: Loaded {len(self.image_ids)} images.")

    def __getitem__(self, index):
        img_id = str(self.image_ids[index])

        # Load FC Feature
        fc_path = os.path.join(self.input_fc_dir, f"{img_id}.npy")
        try:
            fc_feat = np.load(fc_path).astype(np.float32)
        except FileNotFoundError:
            print(f"Missing FC: {img_id}")
            fc_feat = np.zeros((2048,), dtype=np.float32)

        # Load Attention Feature
        att_path = os.path.join(self.input_att_dir, f"{img_id}.npz")
        try:
            att_feat = np.load(att_path)['feat']

            # Reshape về (N, 2048) nếu cần. ResNet thường ra (196, 2048)
            att_feat = att_feat.reshape(-1, att_feat.shape[-1]).astype(np.float32)
        except (FileNotFoundError, KeyError):
            print(f"Missing/Error Att: {img_id}")
            att_feat = np.zeros((1, 2048), dtype=np.float32)

        curr_att_len = att_feat.shape[0]
        final_att_feat = np.zeros([self.max_att_size, att_feat.shape[1]], dtype=np.float32)
        att_mask = np.zeros([self.max_att_size], dtype=np.float32)

        end_idx = min(curr_att_len, self.max_att_size)
        final_att_feat[:end_idx] = att_feat[:end_idx]
        att_mask[:end_idx] = 1  # 1 là vùng có ảnh, 0 là padding

        return {
            'fc_feats': torch.from_numpy(fc_feat),  # Shape: [2048]
            'att_feats': torch.from_numpy(final_att_feat),  # Shape: [196, 2048]
            'att_masks': torch.from_numpy(att_mask),  # Shape: [196]
            'image_ids': img_id  # String ID để tracking
        }

    def __len__(self):
        return len(self.image_ids)


class CaptionDataset(ImageDataset):
    """
    Class mở rộng: Load thêm Caption Labels (Dùng cho Training).
    """

    def __init__(self, split, args):
        super(CaptionDataset, self).__init__(split, args)

        label_path = f'data/processed/{split}_labels.npy'
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Không tìm thấy labels tại: {label_path}")

        self.labels = np.load(label_path)

        assert len(self.labels) == len(self.image_ids), \
            f"Lệch data, Ảnh: {len(self.image_ids)}, Labels: {len(self.labels)}"

    def __getitem__(self, index):
        item = super(CaptionDataset, self).__getitem__(index)
        # Lấy thêm label
        label = self.labels[index]
        item['labels'] = torch.from_numpy(label).long()  # Shape: [max_seq_len] (VD: 20)

        return item