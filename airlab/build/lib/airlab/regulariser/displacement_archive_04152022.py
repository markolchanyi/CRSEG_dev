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
from .. import transformation as T

import numpy as np
import math

from scipy import ndimage
import airlab as al

# Regulariser base class (standard from PyTorch)
class _Regulariser(th.nn.modules.Module):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(_Regulariser, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._weight = 1
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self.name = "parent"
        self._mask = None

    def SetWeight(self, weight):
        print("SetWeight is deprecated. Use set_weight instead.")
        self.set_weight(weight)

    def set_weight(self, weight):
        self._weight = weight

    def set_mask(self, mask):
        self._mask = mask

    def _mask_2d(self, df):
        if not self._mask is None:
            nx, ny, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny].unsqueeze(-1).repeat(1,1,d)
        else:
            return df

    def _mask_3d(self, df, relax_mask=None):
        if not relax_mask is None:
            nx, ny, nz, d = df.shape
            return df * relax_mask.squeeze()[:nx,:ny,:nz].unsqueeze(-1).repeat(1,1,1,d)
        else:
            return df

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return self._weight*tensor.mean()
        if not self._size_average and self._reduce:
            return self._weight*tensor.sum()
        if not self._reduce:
            return self._weight*tensor

"""
    Isotropic TV regularisation
"""
class IsotropicTVRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(IsotropicTVRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "isoTV"

        if self._dim == 2:
            self._regulariser = self._isotropic_TV_regulariser_2d # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._isotropic_TV_regulariser_3d # 3d regularisation

    def _isotropic_TV_regulariser_2d(self, displacement):
        dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2)*self._pixel_spacing[0]
        dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2)*self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0,1,0,1)))

    def _isotropic_TV_regulariser_3d(self, displacement):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2)*self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2)*self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2)*self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0,1,0,1,0,1)))

    def forward(self, displacement):

        # set the supgradient to zeros
        value = self._regulariser(displacement)
        mask = value > 0
        value[mask] = th.sqrt(value[mask])

        return self.return_loss(value)

"""
    TV regularisation
"""
class TVRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(TVRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "TV"

        if self._dim == 2:
            self._regulariser = self._TV_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._TV_regulariser_3d  # 3d regularisation

    def _TV_regulariser_2d(self, displacement):
        dx = th.abs(displacement[1:, 1:, :] - displacement[:-1, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, :] - displacement[1:, :-1, :])*self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _TV_regulariser_3d(self, displacement):
        dx = th.abs(displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :])*self._pixel_spacing[1]
        dz = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :])*self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)))

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))





"""
    Relaxed Diffusion regularisation
"""
class Relaxed_DiffusionRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, mask1, mask2, alpha=0.1, beta=2.7, size_average=True, reduce=True, using_masks=True):
        super(Relaxed_DiffusionRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "Diffusion_relaxed"
        self._mask1 = mask1
        self._mask2 = mask2

        self._alpha = alpha
        self._beta = beta


        self._using_mask = using_masks

        if self._dim == 2:
            self._regulariser = self._l2_regulariser_3d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._l2_regulariser_3d  # 3d regularisation

    def _TV_regulariser_2d(self, displacement):
        dx = th.abs(displacement[1:, 1:, :] - displacement[:-1, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, :] - displacement[1:, :-1, :])*self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _TV_regulariser_3d(self, displacement,hold_mask):
        dx = th.abs(displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :])*self._pixel_spacing[1]
        dz = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :])*self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)), hold_mask)

    def _l2_regulariser_2d(self, displacement):
        dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2) * self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _l2_regulariser_3d(self, displacement,hold_mask):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2) * self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2) * self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)), hold_mask)



    def _vectorized_soft(self, m, alpha, beta,shift=1.0):
        print("vectorizing")
        return 1 + (1-alpha)*(np.square(np.tanh(beta*(m-shift)))-1)

    def _return_matted_transform(self, vol, alpha, beta, use_mask):
        if use_mask:
            #vol_blurred = ndimage.gaussian_filter(vol, sigma=0.2)
            norm_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(vol), sigma=0.2)
            inv_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(1 - vol), sigma=0.2)
            return self._vectorized_soft(norm_transform + inv_transform, alpha, beta)
        else:
            return vol

    def _smoothmax(self,vol1,vol2,alpha=0.01):
        vol_out = np.zeros_like(vol1)
        for i in range(0,vol1.shape[0]):
            for j in range(0,vol1.shape[1]):
                for k in range(0,vol1.shape[2]):
                    p1 = vol1[i,j,k]
                    p2 = vol2[i,j,k]
                    vol_out[i,j,k] = (p1*math.exp(alpha*p1) + p2*math.exp(alpha*p2))/(math.exp(alpha*p1) + math.exp(alpha*p2))
        return vol_out


    def forward(self, displacement):

        # compute displacement field
        grid1 = T.utils.compute_grid(self._mask1.size, dtype=self._mask1.dtype, device=self._mask1.device)
        displacement_upd1 = grid1 + displacement

        grid2 = T.utils.compute_grid(self._mask2.size, dtype=self._mask2.dtype, device=self._mask2.device)
        displacement_upd2 = grid2 + displacement

        warped_mask1 = F.grid_sample(self._mask1.image, displacement_upd1)
        warped_mask2 = F.grid_sample(self._mask2.image, displacement_upd2)

        [s0, s1, s2] = warped_mask1.squeeze().detach().numpy().shape

        warped_mask_np1 = warped_mask1.squeeze().detach().numpy()
        warped_mask_np2 = warped_mask2.squeeze().detach().numpy()

        mask_matted1 = self._return_matted_transform(warped_mask_np1, self._alpha, self._beta, self._using_mask)
        mask_matted2 = self._return_matted_transform(warped_mask_np2, self._alpha, self._beta, self._using_mask)

        smax_mask = self._smoothmax(mask_matted1,mask_matted2,alpha=0.01)

        #smask_matted_torch = th.from_numpy(mask_matted2)
        smask_matted_torch = th.from_numpy(smax_mask)

        #mask_matted = al.Image(mask_matted, [s0, s1, s2], self._pixel_spacing, warped_mask.image.origin, dtype=warped_mask.image.dtype)

        if self._using_mask:
            return self.return_loss(self._regulariser(displacement,smask_matted_torch))
        else:
            return self.return_loss(self._regulariser(displacement))







"""
    Diffusion regularisation
"""
class DiffusionRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(DiffusionRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "L2"

        if self._dim == 2:
            self._regulariser = self._l2_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._l2_regulariser_3d  # 3d regularisation

    def _l2_regulariser_2d(self, displacement):
        dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2) * self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _l2_regulariser_3d(self, displacement):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2) * self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2) * self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)))

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))


"""
    Sparsity regularisation
"""
class SparsityRegulariser(_Regulariser):
    def __init__(self, size_average=True, reduce=True):
        super(SparsityRegulariser, self).__init__([0], size_average, reduce)

    def forward(self, displacement):
        return self.return_loss(th.abs(displacement))
