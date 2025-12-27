import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


# IMAGE_DIRS = [
#     'data/raw/ktvic/',
#     'data/raw/cocovn/'
# ]
IMAGE_DIRS = [
    'data/test_custom/images/',
]

OUTPUT_FC_DIR = 'data/features/fc/'
OUTPUT_ATT_DIR = 'data/features/att/'


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.resnet(images)
        att_feat = features.permute(0, 2, 3, 1)
        fc_feat = self.avgpool(features).squeeze()

        return fc_feat, att_feat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    if not os.path.exists(OUTPUT_FC_DIR): os.makedirs(OUTPUT_FC_DIR)
    if not os.path.exists(OUTPUT_ATT_DIR): os.makedirs(OUTPUT_ATT_DIR)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Load model
    model = EncoderCNN().to(device)
    model.eval()

    for img_dir in IMAGE_DIRS:
        print(f"Processing folder: {img_dir}")
        if not os.path.exists(img_dir):
            print(f"Warning: Folder {img_dir} not found. Skipping.")
            continue

        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for img_name in tqdm(image_files):
            img_path = os.path.join(img_dir, img_name)
            img_id = os.path.splitext(img_name)[0]

            if os.path.exists(os.path.join(OUTPUT_FC_DIR, f"{img_id}.npy")):
                continue

            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    fc, att = model(image)

                np.save(os.path.join(OUTPUT_FC_DIR, f"{img_id}.npy"), fc.cpu().numpy())

                att_cpu = att.cpu().squeeze(0)  # (7, 7, 2048)
                att_reshaped = att_cpu.reshape(-1, 2048).numpy()  # (49, 2048)

                np.savez_compressed(os.path.join(OUTPUT_ATT_DIR, f"{img_id}.npz"), feat=att_reshaped)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print("Extraction completed")


if __name__ == '__main__':
    main()