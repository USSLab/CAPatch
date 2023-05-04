# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np
from torch.functional import Tensor
from torchvision.ops import nms
from torchvision import transforms
from attack_utils import attack
import random
import torch.nn.functional as F
import torch.nn as nn
import math

sys.path.append('detectron2')
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from utils.extract_utils import save_bbox, save_roi_features_by_bbox, save_roi_features
# from utils.progress_bar import ProgressBar
from models import add_config


TEST_SCALES = 600
TEST_MAX_SIZE = 1000

def ColorJitter(patch, brightness = 0, contrast = 0):
    fn_idx = torch.randperm(2)
    brightness_factor = torch.tensor(1.0).uniform_(1-brightness, 1+brightness).item()
    contrast_factor = torch.tensor(1.0).uniform_(1-contrast, 1+contrast).item()
    for fn_id in fn_idx:
        if fn_id == 0:
            patch = (brightness_factor * patch).clamp(0, 1)
        if fn_id == 1:
            tmp = torch.mean((patch[0] * 0.2989 + patch[1] * 0.587 + patch[2] * 0.114).expand(patch.shape), dim=-3, keepdim=True)
            tmp = torch.mean(tmp, dim=-2,  keepdim=True)
            mean = torch.mean(tmp, dim=-1,  keepdim=True)
            patch = (contrast_factor * patch + (1.0 - contrast_factor) * mean).clamp(0, 1)
    return patch

def Robustness(patch,patch_size, h, w,s,angle = math.pi/18, scale=0.8):
    # imga = patch.unsqueeze(0) 
    imga = ColorJitter(patch,brightness=0.15, contrast=0.15).unsqueeze(0) 
    # imga = color_model.color_correction(imga, patch.device) 
    angle = torch.FloatTensor(1).uniform_(-angle, angle)
    angle = angle.to(patch.device)
    scale = torch.FloatTensor(1).fill_(float(s) / patch_size)
    scale = scale.to(patch.device)
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    theta = torch.FloatTensor(1, 2, 3).fill_(0).to(patch.device)
    theta[:, 0, 0] = cos/scale
    theta[:, 0, 1] = sin/scale
    theta[:, 0, 2] = 0
    theta[:, 1, 0] = -sin/scale
    theta[:, 1, 1] = cos/scale
    theta[:, 1, 2] = 0
    
    size = torch.Size((1, 3, patch_size, patch_size))
    grid = F.affine_grid(theta, size)
    output = F.grid_sample(imga, grid)

    rotate_mask = torch.FloatTensor(1, 3, patch_size, patch_size).fill_(1)
    rotate_mask = rotate_mask.to(patch.device)
    output_m = F.grid_sample(rotate_mask, grid)

    pad = nn.ZeroPad2d(
        padding=(w, 600-patch_size-w,
                    h, 600-patch_size-h,)
    )
    mask = pad(output_m)
    paddingimg = pad(output)
    return 1-mask, paddingimg

def tv_loss(img):
    h, w = img.shape[-2], img.shape[-1]
    img_a = img[..., : h - 1, : w - 1]
    img_b = img[..., 1:, : w - 1]
    img_c = img[..., : h - 1, 1:]
    tv = ((img_a - img_b) ** 2 + (img_a - img_c) ** 2 + 1e-9) ** 0.5
    return tv.mean()

def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd

def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def generate_npz(extract_mode, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    else:
        print('Invalid Extract Mode! ')

def getfeat(dataset_dict, model, cfg, im_scale):
    (boxes, scores, features_pooled, attr_scores), rpn_score = model([_ for _ in dataset_dict])
    boxes = [box.tensor for box in boxes] 
    scores = [score for score in scores]
    features_pooled = [feat for feat in features_pooled]
    image_feat=[]
    image_bboxes=[]
    max_confis = []
    for (box, score,feature) in zip(boxes, scores, features_pooled):
        dets = box / dataset_dict[0]['im_scale']
        feats = feature
        max_conf = torch.zeros((score.shape[0])).to(score.device)
        for cls_ind in range(1, score.shape[1]):
            cls_scores = score[:, cls_ind]
            keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                            cls_scores[keep],
                                            max_conf[keep])
        MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES 
        MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
        CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH
        keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
        image_feat.append(feats[keep_boxes])
        image_bboxes.append(dets[keep_boxes])
        max_confis.append(max_conf[keep_boxes])
    return image_feat, image_bboxes, rpn_score, max_confis

def bbox_iou2(box1, box2):

    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)

    iou = inter_area / (b2_area)

    return iou



def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="config/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int, 
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0,1', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='10,100', type=str, 
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="/home/zsb/coco2014/kar_val/")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    cfg = setup(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Extract features.
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()
    attacker = attack()
    batch_size = 2

    center_size = 150
    eot = 30
    patch_size = (center_size + eot ) / 0.8
    patch_size = int(patch_size)

    patch = torch.ones([3, patch_size, patch_size]) * 0.5
    patch.requires_grad_()
    im_scale = 1
    random.seed(1)
    
    for _ in range(1):
        loss_all = []
        for i,im_file in enumerate(imglist[0:30]):
            im = cv2.imread(os.path.join(args.image_dir, im_file))
            x = min(im.shape[0],im.shape[1])
            transform1 = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(x),
                transforms.Resize(600),
            ])
            im = transform1(im)
            if(i % batch_size == 0):
                img_batch =  im.unsqueeze(0)
            else:
                img_batch = torch.cat([img_batch, im.unsqueeze(0)], 0)
            if(i % batch_size == batch_size - 1): 
                s = random.randint(int(patch_size*0.8-2*eot),int(patch_size*0.8))
                h = random.randint(0, 600 - s)
                w = random.randint(0, 600 - s)
                mask, patch_padding =  Robustness(patch,patch_size, h, w,s,angle = math.pi/18, scale=0.8)
                img_batch_patched = (img_batch * mask + patch_padding)*255
                transform2 = transforms.Compose([transforms.Normalize([102.9801, 115.9465, 122.7717], [1, 1, 1]),])
                img_batch_patched = transform2(img_batch_patched)
                dataset_dict = []
                for nu in range(img_batch_patched.shape[0]):
                    dataset_dict_part = {}
                    dataset_dict_part["image"] = img_batch_patched[nu]
                    dataset_dict_part["im_scale"] = im_scale
                    dataset_dict.append(dataset_dict_part)
                image_feat, image_bboxes, rpn_score, max_confis = getfeat(dataset_dict, model, cfg, im_scale)
                att_box = []
                iou = []
                for image_bboxs in image_bboxes:
                    patch_bbox = torch.tensor([w,h,w+s,h+s]).unsqueeze(0).repeat((image_bboxs.shape[0],1)).to(image_bboxs.device)
                    iou2 = bbox_iou2(image_bboxs, patch_bbox)
                    att_box.append(iou2>=0.7)
                    iou.append(iou2)
                loss_cap, loss_att = attacker.batch_loss(image_feat, image_bboxes, att_box)
                loss_det1 = rpn_score
                loss_det2 = (iou[0] * att_box[0]).mean() + (iou[1] * att_box[1]).mean()
                loss_det3 = (max_confis[0] * att_box[0]).mean() + (max_confis[1] * att_box[1]).mean()
                loss_det = -loss_det1/10000 - 0.1 * loss_det2 - 0.1 * loss_det3
                loss_tv = max(tv_loss(patch), 0.04)
                loss = loss_cap + loss_det + 0.1 * loss_att + 2 * loss_tv

                loss_all.append(loss_cap.detach().cpu())
                loss.backward()

                if ( _ <= 1):
                    patch.data -= 0.008 * patch.grad.sign()
                elif ( _ <= 3):
                    patch.data -= 0.004 * patch.grad.sign()
                elif ( _ <= 5):
                # if ( _ <= 1):
                    patch.data -= 0.002 * patch.grad.sign()
                elif ( _ <= 7):
                    patch.data -= 0.001 * patch.grad.sign()
                else:
                    patch.data -= 0.0005 * patch.grad.sign()
                patch.data.clamp_(min=0.0, max=1.0)
                patch.grad.zero_()
            if(i % 20 == 0):
                print(i)
                print(np.array(loss_all).mean())  
        succ = patch.permute([1,2,0]).data.cpu().numpy()*255
        cv2.imwrite('test_out.png',succ)                      


if __name__ == "__main__":
    main()
