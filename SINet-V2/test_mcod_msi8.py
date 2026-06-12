import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.Network_Res2Net_GRA_NCD import Network
from utils.mcod_msi8_dataset import test_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_dir", type=str, default="../MCOD_processed/test/all8_input")
    parser.add_argument("--test_mask_dir", type=str, default="../MCOD_processed/test/ground_truth_mask")
    parser.add_argument("--pth_path", type=str, default="./snapshot/SINet_V2_MCOD_MSI8/Net_epoch_best.pth")
    parser.add_argument("--save_path", type=str, default="./outputs/predictions/sinetv2_msi8")
    parser.add_argument("--trainsize", type=int, default=352)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    model = Network(imagenet_pretrained=False, in_channels=8)
    model.load_state_dict(torch.load(opt.pth_path))
    print("checkpoint loaded path: {}".format(opt.pth_path))
    model.cuda()
    model.eval()

    test_loader = test_dataset(opt.test_img_dir, opt.test_mask_dir, opt.trainsize)

    for _ in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        pred = Image.fromarray((res * 255).astype(np.uint8))
        pred.save(os.path.join(opt.save_path, name))
        print("> MCOD_MSI8 - {}".format(name))
