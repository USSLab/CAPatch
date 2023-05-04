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
from torch.functional import Tensor
from torchvision.ops import nms
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from attack_utils import attack
import random
import math
sys.path.append('detectron2')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from utils.extract_utils import save_bbox, save_roi_features_by_bbox, save_roi_features
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
    imga = ColorJitter(patch,brightness=0.05, contrast=0.05).unsqueeze(0) 
    # imga = color_model.color_correction(imga, patch.device) 
    angle = torch.FloatTensor(1).uniform_(0, 0)
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
    (boxes, scores, features_pooled, attr_scores),_ = model([_ for _ in dataset_dict])
    boxes = [box.tensor for box in boxes] 
    scores = [score for score in scores]
    features_pooled = [feat for feat in features_pooled]
    image_feat=[]
    image_bboxes=[]
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
    return image_feat, image_bboxes


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
                        default="dataset")
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
    num_gpus = len(args.gpu_id.split(','))



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

    tranform = transforms.ToTensor()
    patch = cv2.imread('test_out.png')
    patch = tranform(patch)
    s = 150
    im_scale = 1

    alim = 0.
    succ = 0.
    random.seed(1)
    for i,im_file in enumerate(imglist):
        print(i)
        # if(i < 100):
        #     break
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        x = min(im.shape[0],im.shape[1])
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(x),
            transforms.Resize(600)])
        im = transform1(im)
        if(i % batch_size == 0):
            img_batch =  im.unsqueeze(0)
        else:
            img_batch = torch.cat([img_batch, im.unsqueeze(0)], 0)
        if(i % batch_size == batch_size - 1):                
            h = random.randint(0, 600 - patch_size)
            w = random.randint(0, 600 - patch_size)
            with torch.no_grad():
                mask, patch_padding =  Robustness(patch,patch_size, h, w,s,angle = math.pi/18, scale=0.8)
                img_batch_patched = (mask * img_batch + patch_padding) * 255

                transform2 = transforms.Compose([transforms.Normalize([102.9801, 115.9465, 122.7717], [1, 1, 1]),])
                img_batch_patched = transform2(img_batch_patched)
                dataset_dict = []
                for nu in range(img_batch_patched.shape[0]):
                    dataset_dict_part = {}
                    dataset_dict_part["image"] = img_batch_patched[nu]
                    dataset_dict_part["im_scale"] = im_scale
                    dataset_dict.append(dataset_dict_part)
                image_feat, image_bboxes = getfeat(dataset_dict, model, cfg, im_scale)
                seq,_ = attacker.batch_talk(image_feat, image_bboxes)
                print(seq)
                s_ob = torch.tensor([ 7961, 2784, 3630, 7852, 8912, 7961, 8704, 4437, 9031, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], device=seq.device)
                # a bird is flying over a body of water
                acc = (seq == s_ob).sum(1)
                for _ in acc:
                    alim = alim +1
                    if _ == 20:
                        succ = succ + 1
            print(succ/alim)

   


if __name__ == "__main__":
    main()
