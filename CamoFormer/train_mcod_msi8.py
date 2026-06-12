import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset
from mcod_msi8_dataset import MCODMSI8Dataset
from model.CamoFormer_MSI8 import CamoFormerMSI8


def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return bce + iou.mean()


def resolve_checkpoint_path(path):
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    pth_path = checkpoint_path.with_suffix(".pth")
    if pth_path.is_file():
        return pth_path
    raise FileNotFoundError(f"Checkpoint not found as file or .pth: {path}")


def load_inner_camoformer(model, checkpoint_path, device):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    inner_state = model.camoformer.state_dict()
    load_state = {}
    skipped = []

    for key, value in state.items():
        clean_key = key.replace("module.", "", 1)
        if clean_key in inner_state and inner_state[clean_key].shape == value.shape:
            load_state[clean_key] = value
        else:
            skipped.append(clean_key)

    missing = sorted(set(inner_state) - set(load_state))
    inner_state.update(load_state)
    model.camoformer.load_state_dict(inner_state)

    return checkpoint_path, skipped, missing


def load_encoder_checkpoint(model, checkpoint_path, device):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    encoder_state = model.camoformer.encoder.state_dict()
    load_state = {}
    skipped = []

    for key, value in state.items():
        clean_key = key.replace("module.", "", 1)
        if clean_key in encoder_state and encoder_state[clean_key].shape == value.shape:
            load_state[clean_key] = value
        else:
            skipped.append(clean_key)

    missing = sorted(set(encoder_state) - set(load_state))
    encoder_state.update(load_state)
    model.camoformer.encoder.load_state_dict(encoder_state)

    return checkpoint_path, skipped, missing


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device).float()
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images, images.shape[2:])
        loss = sum(structure_loss(pred, masks) for pred in outputs) / len(outputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_checkpoint(model, save_dir, filename):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CamoFormer_MSI8 on MCOD 8-channel data.")
    parser.add_argument("--train_img_dir", default="../MCOD_processed/train/all8_input")
    parser.add_argument("--train_mask_dir", default="../MCOD_processed/train/ground_truth_mask")
    parser.add_argument("--save_dir", default="checkpoint/CamoFormer_MCOD_MSI8")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--trainsize", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--load", default="checkpoint/CamoFormer-trained")
    parser.add_argument("--encoder_load", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dataset.Config(datapath=args.train_img_dir, snapshot=None, mode="train")
    train_data = MCODMSI8Dataset(args.train_img_dir, args.train_mask_dir, args.trainsize, train=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"number of training samples: {len(train_data)}")
    first_images, first_masks, _ = next(iter(train_loader))
    print(f"first batch input tensor shape: image {tuple(first_images.shape)}, mask {tuple(first_masks.shape)}")

    model = CamoFormerMSI8(cfg).to(device)
    if args.encoder_load:
        loaded_path, skipped, missing = load_encoder_checkpoint(model, args.encoder_load, device)
        print(f"encoder checkpoint path loaded: {loaded_path}")
        print(f"skipped encoder checkpoint keys: {skipped}")
        print(f"missing encoder keys: {missing}")
    elif args.load:
        loaded_path, skipped, missing = load_inner_camoformer(model, args.load, device)
        print(f"checkpoint path loaded: {loaded_path}")
        print(f"skipped checkpoint keys: {skipped}")
        print(f"missing inner CamoFormer keys: {missing}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    for epoch in range(1, args.epoch + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_checkpoint(model, args.save_dir, "Net_epoch_best.pth")
            print(f"best checkpoint save path: {best_path}")

        print(f"Epoch [{epoch:03d}/{args.epoch:03d}] loss: {avg_loss:.6f} best: {best_loss:.6f}")


if __name__ == "__main__":
    main()
