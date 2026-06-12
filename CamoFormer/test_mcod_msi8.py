import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
from mcod_msi8_dataset import MCODMSI8Dataset
from model.CamoFormer_MSI8 import CamoFormerMSI8


def parse_args():
    parser = argparse.ArgumentParser(description="Test CamoFormer_MSI8 on MCOD 8-channel data.")
    parser.add_argument("--test_img_dir", default="../MCOD_processed/test/all8_input")
    parser.add_argument("--test_mask_dir", default="../MCOD_processed/test/ground_truth_mask")
    parser.add_argument("--pth_path", default="checkpoint/CamoFormer_MCOD_MSI8/Net_epoch_best.pth")
    parser.add_argument("--save_path", default="outputs/predictions/camoformer_msi8")
    parser.add_argument("--trainsize", type=int, default=384)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    cfg = dataset.Config(datapath=args.test_img_dir, snapshot=None, mode="test")
    model = CamoFormerMSI8(cfg).to(device)
    model.load_state_dict(torch.load(args.pth_path, map_location=device))
    model.eval()
    print(f"checkpoint path loaded: {args.pth_path}")
    print(f"output prediction folder: {args.save_path}")

    test_data = MCODMSI8Dataset(args.test_img_dir, args.test_mask_dir, args.trainsize, train=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        for images, _, stems in test_loader:
            images = images.to(device).float()
            outputs = model(images, images.shape[2:])
            pred = torch.sigmoid(outputs[-1][0, 0]).cpu().numpy()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            save_file = Path(args.save_path) / f"{stems[0]}.png"
            cv2.imwrite(str(save_file), np.round(pred * 255).astype(np.uint8))
            print(f"> MCOD_MSI8 - {save_file.name}")


if __name__ == "__main__":
    main()
