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


from utils.contours.morph_snakes import morphological_geodesic_active_contour
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import multiprocessing as mp
from utils.contours.cutils import seg2edges, update_callback_in_image, compute_h_additive
from utils.contours.ContourBox import LevelSetAlignmentBase
import torch
import warnings

# this is a workaround to avoid creating/copying the two big arrays to a shared memory area for multi-cpu.
# it only works on POSIX-compliant OS (linux,OSX) and it assumes there is no need for a lock.
# this data is read only.
shared_mem_data = None


class MLS(LevelSetAlignmentBase):
    def _fill_inside(self, bdry, method='fill_holes'):

        if method == 'fill_holes':
            seg = binary_fill_holes(bdry)
            return seg
        else:
            raise ValueError('_fill_inside wrong method:%s' % method)

    def _eval_singleK(self, gt_K, pK_Image, step_ckpts, lambda_, alpha, smoothing,
                      render_radius, is_gt_semantic, **kwargs):

        # This way of checking mostly cares about speed. as I am assuming the whole GT is very sparse.

        all_zeros = not np.any(gt_K)
        # nothing to do
        if all_zeros:
            return gt_K

        if is_gt_semantic is False:
            # filling inside_ to represent the curve
            # this may have problem with boundaries that are not closed(corners of the image dimension)
            # be careful!!
            init_ls = self._fill_inside(gt_K)
        else:
            init_ls = gt_K
            gt_K = seg2edges(gt_K, radius=render_radius)

        h = self._compute_h(gt_K, pK_Image, lambda_, alpha)

        if 'balloon' in kwargs:
            balloon = kwargs['balloon']
        else:
            balloon = 0

        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0

        evolution = morphological_geodesic_active_contour(h, step_ckpts, init_ls, smoothing=smoothing, balloon=balloon, threshold=threshold)

        return evolution

    def process_batch_fn(self, args):
        i, K, gt, pk = args
        gt = gt[i]
        pk = pk[i]

        gt_hat = []

        for j in range(K):
            gtk = gt[j]
            gt_hat_k = self._eval_singleK(gtk, pk[j], **self.options_dict)
            gt_hat.append(gt_hat_k)

        return np.stack(gt_hat, axis=0)  # KxHxW

    def process_batch_hack_multicpu(self, args):
        i, K, mem_id_gt, mem_id_pk, = args

        if shared_mem_data is None:
            raise ValueError()

        gt, pk = shared_mem_data[0], shared_mem_data[1]

        if mem_id_gt != id(gt):
            raise ValueError('error seems the memory id are not the same, not shared array... is this linux?')

        if mem_id_pk != id(pk):
            raise ValueError('error seems the memory id are not the same, not shared array... is this linux?')

        # seems all is fine...let's do it.
        args2 = (i, K, gt, pk)
        return self.process_batch_fn(args2)

    def _multi_cpu_call(self, gt, pk):
        assert gt.shape == pk.shape
        global shared_mem_data
        shared_mem_data = (gt, pk)
        N, K, H, W = pk.shape
        #
        pool = mp.Pool(min(N, self.n_workers))

        output_ = pool.map(self.process_batch_hack_multicpu, [(i, K, id(gt), id(pk)) for i in range(N)])

        pool.close()
        pool.join()

        return output_

    def _multi_cpu_call_2(self, gt, pk):
        """
        -->...similar to above but the workers can be reused in N*K elements instead of just K
        :param gt:
        :param pk:
        :return:
        """

        assert gt.shape == pk.shape
        N, K, H, W = pk.shape

        gt = np.reshape(gt, [N * K, 1, H, W])
        pk = np.reshape(pk, [N * K, 1, H, W])

        global shared_mem_data
        shared_mem_data = (gt, pk)

        #
        pool = mp.Pool(min(N * K, self.n_workers))

        output_ = pool.map(self.process_batch_hack_multicpu,
                           [(i, 1, id(gt), id(pk)) for i in range(N * K)])

        pool.close()
        pool.join()
        # we need to reorder the array to make it compatible with the rest of the api.

        return output_

    def __call__(self, gt_dict, pk):

        if not isinstance(gt_dict, dict):
            raise ValueError('wrong param')

        gt = gt_dict['seg']
        if isinstance(gt, torch.Tensor):
            gt = gt.byte().cpu().float().numpy()

        if isinstance(pk, torch.Tensor):
            pk = pk.cpu().numpy()

        assert gt.shape == pk.shape
        N, K, H, W = pk.shape

        if N * K > 1 and self.n_workers > 1:
            output_ = self._multi_cpu_call_2(gt, pk)
        else:
            output_ = []
            for i in range(N):
                gt_hat = self.process_batch_fn((i, K, gt, pk))
                output_.append(gt_hat)

        return np.stack(output_, axis=0) # NxKxLStepsxHxW
