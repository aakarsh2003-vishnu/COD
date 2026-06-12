import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset
from model.CamoFormer import CamoFormer


def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return bce + iou.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
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
    torch.save(model.state_dict(), os.path.join(save_dir, filename))


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CamoFormer on MCOD false-colour data.")
    parser.add_argument("--train_img_dir", default="../MCOD_processed/train/official_false_colour")
    parser.add_argument("--train_mask_dir", default="../MCOD_processed/train/ground_truth_mask")
    parser.add_argument("--save_dir", default="checkpoint/CamoFormer_MCOD")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--trainsize", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--load", default="checkpoint/CamoFormer-trained.pth")
    parser.add_argument("--encoder_load", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dataset.Config(
        datapath=args.train_img_dir,
        image_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        snapshot=None,
        mode="train",
        trainsize=args.trainsize,
    )

    train_data = dataset.Data(cfg, "CamoFormer")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=8,
        collate_fn=train_data.collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = CamoFormer(cfg, load_path=args.encoder_load).to(device)

    if args.load:
        state = torch.load(args.load, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded full CamoFormer checkpoint: {args.load}")
    elif args.encoder_load:
        print(f"Using ImageNet/pretrained encoder only: {args.encoder_load}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    for epoch in range(1, args.epoch + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, args.save_dir, "Net_epoch_best.pth")
            print(f"Saved best checkpoint at epoch {epoch} with loss {best_loss:.6f}")

        print(f"Epoch [{epoch:03d}/{args.epoch:03d}] loss: {avg_loss:.6f} best: {best_loss:.6f}")

if __name__ == "__main__":
    main()
