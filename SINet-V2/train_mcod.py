import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from lib.Network_Res2Net_GRA_NCD import Network
from utils.mcod_dataset import get_loader, test_dataset
from utils.utils import adjust_lr, clip_gradient


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, total_step, save_dir, writer, opt):
    model.train()
    loss_all = 0
    epoch_step = 0

    for i, (images, gts) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images = images.cuda()
        gts = gts.cuda()

        preds = model(images)
        loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts)
        loss_final = structure_loss(preds[3], gts)
        loss = loss_init + loss_final

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        opt.step += 1
        epoch_step += 1
        loss_all += loss.data

        if i % 20 == 0 or i == total_step or i == 1:
            print("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}".
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
            logging.info(
                "[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}".
                format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data)
            )
            writer.add_scalars(
                "Loss_Statistics",
                {"Loss_init": loss_init.data, "Loss_final": loss_final.data, "Loss_total": loss.data},
                global_step=opt.step,
            )
            grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image("RGB", grid_image, opt.step)
            grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image("GT", grid_image, opt.step)

            res = preds[0][0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            writer.add_image("Pred_init", torch.tensor(res), opt.step, dataformats="HW")
            res = preds[3][0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            writer.add_image("Pred_final", torch.tensor(res), opt.step, dataformats="HW")

    loss_all /= epoch_step
    logging.info("[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}".format(epoch, opt.epoch, loss_all))
    writer.add_scalar("Loss-epoch", loss_all, global_step=epoch)
    if epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, "Net_epoch_{}.pth".format(epoch)))


def val(val_loader, model, epoch, save_dir, writer, opt):
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for _ in range(val_loader.size):
            image, gt, _, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)
            res = F.upsample(res[3], size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / val_loader.size
        writer.add_scalar("MAE", torch.tensor(mae), global_step=epoch)
        print("Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.".format(epoch, mae, opt.best_mae, opt.best_epoch))
        if epoch == 1 or mae < opt.best_mae:
            opt.best_mae = mae
            opt.best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "Net_epoch_best.pth"))
            print("Save state_dict successfully! Best epoch:{}.".format(epoch))
        logging.info("[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}".format(
            epoch, mae, opt.best_epoch, opt.best_mae
        ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_dir", type=str, required=True)
    parser.add_argument("--train_mask_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./snapshot/SINet_V2_MCOD/")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=36)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    model = Network(channel=32, imagenet_pretrained=opt.imagenet_pretrained).cuda()
    if opt.imagenet_pretrained:
        print("Using ImageNet-pretrained Res2Net backbone.")
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print("load model from ", opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("load data...")
    train_loader = get_loader(
        image_root=opt.train_img_dir,
        mask_root=opt.train_mask_dir,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.num_workers,
    )
    val_loader = test_dataset(
        image_root=opt.train_img_dir,
        mask_root=opt.train_mask_dir,
        testsize=opt.trainsize,
    )
    total_step = len(train_loader)

    logging.basicConfig(
        filename=os.path.join(save_dir, "log.log"),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )
    logging.info("Network-Train-MCOD")
    logging.info(
        "Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; "
        "save_dir: {}; decay_epoch: {}".format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_dir, opt.decay_epoch
        )
    )

    opt.step = 0
    opt.best_mae = 1
    opt.best_epoch = 0
    writer = SummaryWriter(os.path.join(save_dir, "summary"))

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar("learning_rate", cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, total_step, save_dir, writer, opt)
        val(val_loader, model, epoch, save_dir, writer, opt)
