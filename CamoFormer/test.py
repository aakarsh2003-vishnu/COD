import os
import sys
import datetime
import math
import dataset
import argparse
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.CamoFormer import CamoFormer 
import matplotlib.pyplot as plt
plt.ion()

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test(dataset,parser):
    args   = parser.parse_args()
    _TESTDATASET_ = args.test_dataset
    _CKPT_ = args.ckpt

    if args.test_img_dir:
        t = Valid(
            dataset,
            args.test_img_dir,
            0,
            'CamoFormer',
            _CKPT_,
            mode='test',
            image_dir=args.test_img_dir,
            mask_dir=args.test_mask_dir,
            save_root=args.save_root,
            dataset_name=args.dataset_name,
        )
        t.save()
        return

    for path in ['CHAMELEON','COD10K','NC4K','CAMO']:
        path=_TESTDATASET_+'/'+path
        t = Valid(dataset, path, 0, 'CamoFormer', _CKPT_, mode='test', save_root=args.save_root)
        t.save()


class Valid(object):
    def __init__(
        self,
        Dataset,
        Path,
        epoch,
        model_name,
        checkpoint_path,
        mode='Valid',
        image_dir=None,
        mask_dir=None,
        save_root='output/Prediction/CamoFormer-test',
        dataset_name=None,
    ):
        ## dataset
        if mode == 'test':
            self.cfg = Dataset.Config(
                datapath=Path,
                image_dir=image_dir,
                mask_dir=mask_dir,
                snapshot=checkpoint_path,
                mode='test',
            )
        else:
            self.cfg = Dataset.Config(datapath=Path, snapshot=checkpoint_path+model_name+str(epoch+1), mode='test')
        self.mode = mode
        self.save_root = save_root
        self.dataset_name = dataset_name or self.cfg.datapath.split('/')[-1]
        self.data   = Dataset.Data(self.cfg, model_name)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = CamoFormer(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.epoch = epoch
        
    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                P5, P4, P3, P2, P1 = self.net(image, shape)
                pred = torch.sigmoid(P1[0,0]).cpu().numpy()*255
                if self.mode == 'test':
                    head = self.save_root + '/' + self.dataset_name
                else:
                    head = self.save_root + '/CamoFormer-epoch'+str(self.epoch+1)+'/'+ self.dataset_name
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+os.path.splitext(name[0])[0]+'.png', np.round(pred))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='CamoFormer')
    parser.add_argument("--test_dataset", default='dataset/TestDataset')
    parser.add_argument("--test_img_dir", default=None)
    parser.add_argument("--test_mask_dir", default=None)
    parser.add_argument("--dataset_name", default='MCOD')
    parser.add_argument("--save_root", default='output/Prediction/CamoFormer-test')
    parser.add_argument("--ckpt", default='CamoFormer.pth')
    args   = parser.parse_args()
    test(dataset, parser)
       
