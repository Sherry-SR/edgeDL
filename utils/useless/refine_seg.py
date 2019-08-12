# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import importlib

import tqdm
from utils.contours import ContourBox
import argparse
import ast
import numpy as np
import os
import nibabel as nib
"""
class GenerateGT_PNGMask:
    def __init__(self, classes_to_keep, output_dir):
        self.classes_to_keep = classes_to_keep  # Object classes: [11, 12, 13, 14, 15, 16, 17, 18]
        self.output_dir = output_dir

    def _save_fname(self, filename):
        city_name = filename.split('_')[0]
        gt_name = filename.split('_leftImg8bit')[0] + 'gtCoarseR_labelIds.png'

    def generate_save(self, gt, improved, filename):
        only_objects_updated, fully_improved = self._generate_single_mask(gt, improved)

        city_name = filename.split('_')[0]
        gt_name_objects = filename.split('_leftImg8bit')[0] + '_gtCoarseRefObj_labelIds.png'
        gt_name_alls = filename.split('_leftImg8bit')[0] + '_gtCoarseRefAll_labelIds.png'
        output_dir = os.path.join(self.output_dir, city_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        gt_name_alls = os.path.join(output_dir, gt_name_alls)

        fully_improved = Image.fromarray(fully_improved)
        fully_improved.save(gt_name_alls, 'png')

        gt_name_objects = os.path.join(output_dir, gt_name_objects)

        only_objects_updated = Image.fromarray(only_objects_updated)
        only_objects_updated.save(gt_name_objects, 'png')

    def _generate_single_mask(self, gt, improved):
        final_canvas = np.zeros(gt.shape[1:]).astype(np.uint8)
        all_updated_canvas = np.zeros(gt.shape[1:]).astype(np.uint8)

        for k, (gt_k, improved_k) in enumerate(zip(gt, improved), start=0):

            if k not in self.classes_to_keep:
                if np.any(gt_k):
                    final_canvas[gt_k != 0] = k
            else:
                if np.any(improved_k) and np.any(gt_k):
                    final_canvas[improved_k != 0] = k

            all_updated_canvas[improved_k != 0] = k

        #

        return final_canvas, all_updated_canvas
"""

def expand_as_one_hot(input, C):
    """
    Converts DxHxW label image to CxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 3D input image (DxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 4D output image (CxDxHxW)
    """
    output = []
    for c in range(1, C+1):
        mask = input == c
        output.append(mask)

    output = np.array(output, dtype = np.int)
    
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', type=str,
                        default='/home/SENSETIME/shenrui/data/pelvis_resampled/dataset_train_temp.txt')

    parser.add_argument('--in_dir', type=str,
                        default='/home/SENSETIME/shenrui/data/pelvis_predict')

    parser.add_argument('--val_file_list', type=str,
                        default='./Cityscapes/benchmark/datadir/val.txt')

    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--n_classes_start', type=int, default=1)
    parser.add_argument('--level_set_method', type=str, default='MLS')
    parser.add_argument('--level_set_config_dict', type=dict, default={})
    # ---
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--smooth_lsteps', type=int, default=1)
    parser.add_argument('--lambda_', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--step_ckpts', type=str, default="[0,60]")
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='/home/SENSETIME/shenrui/data/refined_data_test')
    parser.add_argument('--classes_to_keep', type=list, default=[])
    parser.add_argument('--balloon', type=float, default=0)
    parser.add_argument('--threshold', type=float, default=0.95)

    args = parser.parse_args()

    level_set_config_dict = {
        'lambda_': args.lambda_,
        'alpha': args.alpha,
        'smoothing': args.smooth_lsteps,
        'render_radius': 1,
        'is_gt_semantic': True,
        'method': args.level_set_method,
        'balloon': args.balloon,
        'threshold': args.threshold,
        'step_ckpts': ast.literal_eval(args.step_ckpts)
    }

    args.level_set_config_dict = level_set_config_dict

    return args


def do_it(args):
    in_dir = args.in_dir
    val_file_list = args.val_file_list
    coarse_dir = args.coarse_dir
    n_classes_interval = (args.n_classes_start, args.n_classes)
    level_set_config_dict = args.level_set_config_dict

    
    path_gt = '/home/SENSETIME/shenrui/data/pelvis_resampled/28-ct_label.nii.gz'
    path_pred = '/home/SENSETIME/shenrui/data/pelvis_predict/28-ct_predictions.nii.gz'

    pred_img = nib.load(path_pred)
    pred = np.transpose(pred_img.get_fdata())[:,100:110, :, :]

    gt_img = nib.load(path_gt)
    gt = np.transpose(gt_img.get_fdata())
    gt = expand_as_one_hot(gt, pred.shape[0])[:,100:110, :, :]

    affine = gt_img.affine

    cbox = ContourBox.LevelSetAlignment(n_workers=0, config=level_set_config_dict)

    #mask_generator = GenerateGT_PNGMask(classes_to_keep, args.output_dir)

    output, _ = cbox({'seg': gt, 'bdry': None}, pred)
    improved_mask = np.transpose(output[:, :, -1, :, :])
    original_mask =  np.transpose(output[:, :, 0, :, :])
    nib.save(nib.Nifti1Image(improved_mask, affine), '/home/SENSETIME/shenrui/data/pelvis_predict/28_temp.nii.gz')
    nib.save(nib.Nifti1Image(original_mask, affine), '/home/SENSETIME/shenrui/data/pelvis_predict/28_temp0.nii.gz')
    return improved_mask

if __name__ == "__main__":
    args = parse_args()
    do_it(args)
