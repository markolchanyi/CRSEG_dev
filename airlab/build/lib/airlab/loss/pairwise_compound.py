# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as th
import torch.nn.functional as F
import airlab as al

import numpy as np

from .. import transformation as T
from ..transformation import utils as tu
from ..utils import kernelFunction as utils
from .... import varifold as vf

from packaging import version
from scipy import ndimage




# Loss base class (standard from PyTorch)
class _PairwiseImageLoss_compound(th.nn.modules.Module):
    def __init__(self,
    fixed_image1,
    moving_image1,
    fixed_image2,
    moving_image2,
    fixed_roi,
    fixed_mask1,
    fixed_mask2,
    moving_roi,
    moving_mask1,
    moving_mask2,
    roi_weight,
    mask1_weight,
    mask2_weight,
    channel1_weight,
    channel2_weight,
    fixed_dummymask,
    moving_dummymask,
    epsilon,
    size_average,
    reduce,
    single_channel,
    no_superstructs,
    varifold,
    MSE,
    Dice):
        super(_PairwiseImageLoss_compound, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._name = "parent"

        self._warped_moving_image1 = None
        self._warped_moving_image2 = None
        self._warped_moving_roi = None
        self._warped_moving_mask1 = None
        self._warped_moving_mask2 = None
        self._weight = 1
        self._single_channel = single_channel
        self._no_superstructs = no_superstructs
        self._use_varifold = varifold
        self._use_mse = MSE
        self._use_dice = Dice

        self._fixed_image1 = fixed_image1
        self._fixed_image2 = fixed_image2
        self._moving_image1 = moving_image1
        self._moving_image2 = moving_image2


        self._fixed_roi = fixed_roi
        self._moving_roi = moving_roi
        self._fixed_mask1 = fixed_mask1
        self._moving_mask1 = moving_mask1
        self._fixed_mask2 = fixed_mask2
        self._moving_mask2 = moving_mask2

        self._roi_weight = roi_weight
        self._mask1_weight = mask1_weight
        self._mask2_weight = mask2_weight
        self._channel1_weight = channel1_weight
        self._channel2_weight = channel2_weight

        self._moving_dummymask = moving_dummymask
        self._fixed_dummymask = fixed_dummymask
        self._grid = None


        assert self._moving_image1 != None and self._fixed_image2 != None
        assert self._moving_image1.size == self._fixed_image2.size
        assert self._moving_image1.device == self._fixed_image2.device
        assert len(self._moving_image1.size) == 2 or len(self._moving_image1.size) == 3

        self._grid = T.utils.compute_grid(self._moving_image1.size, dtype=self._moving_image1.dtype,
        device=self._moving_image1.device)

        self._dtype = self._moving_image1.dtype
        self._device = self._moving_image1.device

        @property
        def name(self):
            return self._name

        def GetWarpedImages(self):
            return self._warped_moving_image1[0, 0, ...].detach().cpu(), self._warped_moving_image2[0, 0, ...].detach().cpu()


        def GetCurrentMask(self, displacement, moving, fixed):
            """
            Computes a mask defining if pixels are warped outside the image domain, or if they fall into
            a fixed image mask or a warped moving image mask.
            return (Tensor): maks array
            """
            # exclude points which are transformed outside the image domain
            mask = th.zeros_like(self._fixed_image1.image, dtype=th.uint8, device=self._device)
            for dim in range(displacement.size()[-1]):
                mask += displacement[..., dim].gt(1) + displacement[..., dim].lt(-1)
            mask = mask == 0
            # and exclude points which are masked by the warped moving and the fixed mask
            if not moving is None:
                tmp = F.grid_sample(moving.image, displacement)
                tmp = tmp >= 0.5

                # if either the warped moving mask or the fixed mask is zero take zero,
                # otherwise take the value of mask
                if not fixed is None:
                    mask = th.where(((tmp == 0) | (fixed == 0)), th.zeros_like(mask), mask)
                else:
                    mask = th.where((tmp == 0), th.zeros_like(mask), mask)
            return mask

        def set_loss_weight(self, weight):
            self._weight = weight

        # conditional return
        def return_loss(self, tensor):
            if self._size_average and self._reduce:
                return tensor.mean()*self._weight
            if not self._size_average and self._reduce:
                return tensor.sum()*self._weight
            if not self.reduce:
                return tensor*self._weight






class COMPOUND(_PairwiseImageLoss_compound):
r""" Implementation of a compound Normalized Gradient Field image loss with varifold embedding.
Args:
fixed_image (Image): Fixed image for the registration
moving_image (Image): Moving image for the registration
fixed_mask (Tensor): Mask for the fixed image
moving_mask (Tensor): Mask for the moving image
epsilon (float): Regulariser for the gradient amplitude
size_average (bool): Average loss function
reduce (bool): Reduce loss function to a single value
"""
    def __init__(self,
    fixed_image1,
    moving_image1,
    fixed_image2,
    moving_image2,
    fixed_roi,
    fixed_mask1,
    fixed_mask2,
    moving_roi,
    moving_mask1,
    moving_mask2,
    roi_weight=1.0,
    mask1_weight=1.0,
    mask2_weight=1.0,
    channel1_weight=1.0,
    channel2_weight=1.0,
    fixed_dummymask=None,
    moving_dummymask=None,
    epsilon=1e-5,
    size_average=True,
    reduce=True,
    single_channel=False,
    no_superstructs=False,
    varifold=True,
    generate_mesh=False,
    cts1_fixed=None,
    norms1_fixed=None,
    cts2_fixed=None,
    norms2_fixed=None,
    verts1_moving=None,
    faces1_moving=None,
    verts2_moving=None,
    faces2_moving=None,
    MSE=False,
    Dice=False):
        super(COMPOUND, self).__init__(fixed_image1,
        ixed_image1,
        moving_image1,
        fixed_image2,
        moving_image2,
        fixed_roi,
        fixed_mask1,
        fixed_mask2,
        moving_roi,
        moving_mask1,
        moving_mask2,
        roi_weight,
        mask1_weight,
        mask2_weight,
        channel1_weight,
        channel2_weight,
        fixed_dummymask,
        moving_dummymask,
        epsilon,
        size_average,
        reduce,
        single_channel,
        no_superstructs,
        varifold,
        MSE,
        Dice)

        self._name = "compound_loss"

        self._dim = fixed_image1.ndim
        self._epsilon = epsilon


        if self._dim == 2:
            dx = (fixed_image.image[..., 1:, 1:] - fixed_image.image[..., :-1, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:] - fixed_image.image[..., 1:, :-1]) * fixed_image.spacing[1]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_2d
        else:
            dx = (fixed_image1.image[..., 1:, 1:, 1:] - fixed_image1.image[..., :-1, 1:, 1:]) * fixed_image1.spacing[0]
            dy = (fixed_image1.image[..., 1:, 1:, 1:] - fixed_image1.image[..., 1:, :-1, 1:]) * fixed_image1.spacing[1]
            dz = (fixed_image1.image[..., 1:, 1:, 1:] - fixed_image1.image[..., 1:, 1:, :-1]) * fixed_image1.spacing[2]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy) + th.abs(dz))*0.005

                norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

                self._ng_fixed_image1 = F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

                ###########

                dx = (fixed_image2.image[..., 1:, 1:, 1:] - fixed_image2.image[..., :-1, 1:, 1:]) * fixed_image2.spacing[0]
                dy = (fixed_image2.image[..., 1:, 1:, 1:] - fixed_image2.image[..., 1:, :-1, 1:]) * fixed_image2.spacing[1]
                dz = (fixed_image2.image[..., 1:, 1:, 1:] - fixed_image2.image[..., 1:, 1:, :-1]) * fixed_image2.spacing[2]

                if self._epsilon is None:
                    with th.no_grad():
                        self._epsilon = th.mean(th.abs(dx) + th.abs(dy) + th.abs(dz))*0.005
                        print("epsilon is: ", self._epsilon)

                norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

                self._ng_fixed_image2 = F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))


                self._ngf_loss = self._ngf_loss_3d

    def _retrieve_np_displacement(self, displacement):
        return displacement.detach().cpu().numpy()

    def _update_mesh()

    def _ngf_loss_2d(self, warped_image):
        dx = (warped_image[..., 1:, 1:] - warped_image[..., :-1, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:] - warped_image[..., 1:, :-1]) * self._moving_image.spacing[1]
        norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

    def _ngf_loss_3d(self, warped_image):
        dx = (warped_image[..., 1:, 1:, 1:] - warped_image[..., :-1, 1:, 1:]) * self._moving_image1.spacing[0]
        dy = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, :-1, 1:]) * self._moving_image1.spacing[1]
        dz = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, 1:, :-1]) * self._moving_image1.spacing[2]
        norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        if self._use_varifold:
            disp_np = self._retrieve_np_displacement(displacement)
            self._moving_cts1, self._moving_norms1 = vf.update_mesh(verts1_moving,faces1_moving,displacement)
            self._moving_cts2, self._moving_norms2 = vf.update_mesh(verts2_moving,faces2_moving,displacement)
            self._varifold_loss = self.mask1_weight*vf.varifold_distance_nomesh(self._moving_cts1, self._moving_norms1, cts1_fixed, norms1_fixed, sigma=3)
            self._varifold_loss += self.mask2_weight*vf.varifold_distance_nomesh(self._moving_cts2, self._moving_norms2, cts1_fixed, norms1_fixed, sigma=3)

        # compute current mask and dummy ones mask
        self._loss_region = super(NGF_2CHAN, self).GetCurrentMask_dummy(displacement, self._moving_roi, self._fixed_roi)

        self._warped_moving_image1 = F.grid_sample(self._moving_image1.image, displacement)
        self._warped_moving_image2 = F.grid_sample(self._moving_image2.image, displacement)

        # compute the gradient of the warped image
        ng_warped_image1 = self._ngf_loss(self._warped_moving_image1)
        ng_warped_image2 = self._ngf_loss(self._warped_moving_image2)

        self._moving_roi_warped = F.grid_sample(self._moving_roi,displacement)
        self._moving_mask1_warped = F.grid_sample(self._moving_mask1,displacement)
        self._moving_mask2_warped = F.grid_sample(self._moving_mask2,displacement)

        self._fixed_mask1 = self._fixed_mask1.to(dtype=self._dtype, device=self._device)
        self._fixed_mask2 = self._fixed_mask2.to(dtype=self._dtype, device=self._device)

        value = 0
        for dim in range(self._dim):
            value = value + ng_warped_image1[:, dim, ...] * self._ng_fixed_image1[:, dim, ...] + self._channel_weight * ng_warped_image2[:, dim, ...] * self._ng_fixed_image2[:, dim, ...]

        value = 0.5 * th.masked_select(-value.pow(2), mask).mean() + self._varifold_loss

        return value

        # compute squared differences
        #MSE_val = (warped_m_mask - self._f_mask.image).pow(2)
        #MSE_subval = (warped_m_submask - self._f_submask.image).pow(2)

        #if not self._single_channel:
        #    value = 0
        #for dim in range(self._dim):
        #value = value + ng_warped_image1[:, dim, ...] * self._ng_fixed_image1[:, dim, ...] + self._channel_weight * ng_warped_image2[:, dim, ...] * self._ng_fixed_image2[:, dim, ...]
        #else:
        #value = 0
        #for dim in range(self._dim):
        #value = value + ng_warped_image1[:, dim, ...] * self._ng_fixed_image1[:, dim, ...]

        #if self._no_superstructs:
        #value = 0.5 * th.masked_select(-value.pow(2), mask)
        #else:
        #value = 0.5 * th.masked_select(-value.pow(2), mask) + self._mask_weight * self._multiplier * th.masked_select(MSE_val, mask) + self._submask_weight * self._multiplier * th.masked_select(MSE_subval, mask)

        #return self.return_loss(value)
