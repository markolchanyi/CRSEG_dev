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
    def __init__(self, pixel_spacing, mask_list, alpha=0.1, beta=2.7, size_average=True, reduce=True, using_masks=True,relax_regularization=True):
        super(Relaxed_DiffusionRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "Diffusion_relaxed"
        self._mask_list = mask_list

        self._alpha = alpha
        self._beta = beta
        self._relax_regularization = relax_regularization


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
        return 1 + (1-alpha)*(np.square(np.tanh(beta*(m-shift)))-1)

    def _return_matted_transform(self, vol, alpha, beta, use_mask):
        if use_mask:
            #vol_blurred = ndimage.gaussian_filter(vol, sigma=0.2)
            norm_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(vol), sigma=0.2)
            inv_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(1 - vol), sigma=0.2)
            return self._vectorized_soft(norm_transform + inv_transform, alpha, beta)
        else:
            return vol

    def _smoothmax(self,vol_list,alpha=0.01):

        vol_out = np.zeros_like(vol_list[0])

        for i in range(0,vol_list[0].shape[0]):
            for j in range(0,vol_list[0].shape[1]):
                for k in range(0,vol_list[0].shape[2]):
                    numer=0
                    denom=0
                    for ind in range(0,len(vol_list)):
                        vol=vol_list[ind]
                        p = vol[i,j,k]
                        numer += p*math.exp(alpha*p)
                        denom += math.exp(alpha*p)

                    vol_out[i,j,k] = numer/denom
        return vol_out


    def forward(self, displacement):
        if self._relax_regularization:
            warped_mask_matted_list = []
            for i in range(0,len(self._mask_list)):
                # compute displacement field
                grid1 = T.utils.compute_grid(self._mask_list[i].size, dtype=self._mask_list[i].dtype, device=self._mask_list[i].device)
                displacement_upd = grid1 + displacement

                warped_mask = F.grid_sample(self._mask_list[i].image, displacement_upd)
                warped_mask_np = warped_mask.squeeze().detach().numpy()
                mask_matted = self._return_matted_transform(warped_mask_np, self._alpha, self._beta, self._using_mask)

                warped_mask_matted_list.append(mask_matted)

            smax_mask = self._smoothmax(warped_mask_matted_list,alpha=0.01)
            smask_matted_torch = th.from_numpy(smax_mask)

            if self._using_mask:
                return self.return_loss(self._regulariser(displacement,smask_matted_torch))
            else:
                return self.return_loss(self._regulariser(displacement))
        else:
            return self.return_loss(self._regulariser(displacement,hold_mask=None))





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
