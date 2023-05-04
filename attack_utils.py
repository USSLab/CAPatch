from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from torch.functional import Tensor

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import argparse
import captioning.utils.misc as utils
import torch

class attack():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='data/models/model_up.pth',
                        help='path to model to evaluate')
        parser.add_argument('--cnn_model', type=str,  default='resnet101',
                        help='resnet101, resnet152')
        parser.add_argument('--infos_path', type=str, default='data/models/infos_up.pkl',
                        help='path to infos to evaluate')
        parser.add_argument('--only_lang_eval', type=int, default=0,
                        help='lang eval on saved results')
        parser.add_argument('--force', type=int, default=0,
                        help='force to evaluate no matter if there are results available')
        parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
        opts.add_eval_options(parser)
        opts.add_diversity_opts(parser)
        opt = parser.parse_args(args = '')
        # Load infos
        with open(opt.infos_path, 'rb') as f:       
            infos = utils.pickle_load(f)
        # override and collect parameters
        replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
        ignore = ['start_from']

        for k in vars(infos['opt']).keys():
            if k in replace:
                setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
            elif k not in ignore:
                if not k in vars(opt):
                    vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

        vocab = infos['vocab'] # ix -> word mapping
        opt.vocab = vocab

        self.model = models.setup(opt)
        del opt.vocab
        self.model.load_state_dict(torch.load(opt.model, map_location='cpu'))
        self.model.to(opt.device)
        self.model.eval()

        self.loader = DataLoader(opt)
        self.loader.dataset.ix_to_word = infos['vocab']
        opt.dataset = opt.input_json

        self.eval_kwargs = vars(opt)
        verbose = self.eval_kwargs.get('verbose', True)
        verbose_beam = self.eval_kwargs.get('verbose_beam', 0)
        verbose_loss = self.eval_kwargs.get('verbose_loss', 1)
        num_images = self.eval_kwargs.get('num_images', self.eval_kwargs.get('val_images_use', -1))
        self.split = self.eval_kwargs.get('split', 'val')
        lang_eval = self.eval_kwargs.get('language_eval', 0)
        dataset = self.eval_kwargs.get('dataset', 'coco')
        beam_size = self.eval_kwargs.get('beam_size', 1)
        sample_n = self.eval_kwargs.get('sample_n', 1)
        remove_bad_endings = self.eval_kwargs.get('remove_bad_endings', 0)
        os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
        self.device = self.eval_kwargs.get('device', 'cuda')

        self.n = 0


    def batch_loss(self, image_feat, image_bboxes, box,depe = 0):
        max_n = max(image_feat[0].shape[0], image_feat[1].shape[0])
        att_masks = torch.ones((2,max_n) ,device=self.device)
        att_boxes = torch.zeros((2,max_n) ,device=self.device)
        for _ in range(2):
            for i, j in enumerate(box[_]):
                if(j == True):
                    att_boxes[_][i] = 1
        for _ in range(2):
            if(_ == 0):
                fc_feats = image_feat[_].mean(0)
            else:
                fc_feats = torch.cat((fc_feats.unsqueeze_(0),image_feat[_].mean(0).unsqueeze_(0)),0)
            if(image_feat[_].shape[0] != max_n):
                att_masks[_, image_feat[_].shape[0]:max_n] = 0
                image_feat[_] = torch.cat([image_feat[_],torch.zeros((max_n - image_feat[_].shape[0], 2048)).to(image_feat[_].device)], 0)
        att_feats = torch.cat([_.unsqueeze_(0) for _ in image_feat], 0)

        tmp_eval_kwargs = self.eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        s_ob = torch.tensor([7961, 2784, 3630, 7852, 8912, 7961, 8704, 4437, 9031, 0, 0, 0, 0, 0, 0, 0], device=self.device)
        # a bird is flying over a body of water
        tmp_eval_kwargs['depe'] = depe
        loss, loss_att= self.model(fc_feats, att_feats, s_ob, att_masks, att_boxes, opt=tmp_eval_kwargs, mode='attack')
        print(loss)
        return loss.mean(), -loss_att

    def batch_talk(self, image_feat, image_bboxes):
        max_n = max(image_feat[0].shape[0], image_feat[1].shape[0])
        att_masks = torch.ones((2,max_n) ,device=self.device)

        for _ in range(2):
            if(_ == 0):
                fc_feats = image_feat[_].mean(0)
            else:
                fc_feats = torch.cat((fc_feats.unsqueeze_(0),image_feat[_].mean(0).unsqueeze_(0)),0)
            if(image_feat[_].shape[0] != max_n):
                att_masks[_, image_feat[_].shape[0]:max_n-1] = 0
                image_feat[_] = torch.cat([image_feat[_],torch.zeros((max_n - image_feat[_].shape[0], 2048)).to(image_feat[_].device)], 0)
        att_feats = torch.cat([_.unsqueeze_(0) for _ in image_feat], 0)

        tmp_eval_kwargs = self.eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        seq, _ = self.model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        seq = seq.data
        sents = utils.decode_sequence(self.model.vocab, seq)
        return seq, sents



