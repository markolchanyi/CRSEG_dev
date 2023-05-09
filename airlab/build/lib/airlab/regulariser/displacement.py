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
from torchvision.transforms import GaussianBlur
from .. import transformation as T

import numpy as np
import math

from scipy import ndimage
import airlab as al
from dipy.io.image import save_nifti

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

    def _mask_3d(self, df, relax_mask=None, mask_weight=5):
        if not relax_mask is None:
            nx, ny, nz, d = df.shape
            return df * mask_weight * relax_mask.squeeze()[:nx,:ny,:nz].unsqueeze(-1).repeat(1,1,1,d)
        #else:
        #    nx, ny, nz, d = df.shape
        #    return df + mask_weight*relax_mask.squeeze()[:nx,:ny,:nz].unsqueeze(-1).repeat(1,1,1,d)

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

    def _l2_regulariser_3d(self, displacement, hold_mask):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2) * self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2) * self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)), hold_mask)



    def _vectorized_soft(self, m, alpha, beta, shift=1.0):
        return 1 + (1-alpha)*(np.square(np.tanh(beta*(m-shift)))-1)



    def _return_matted_transform(self, vol, alpha=20.0, beta=1.1, use_mask=True):
        if use_mask:
            #vol_blurred = ndimage.gaussian_filter(vol, sigma=0.2)
            norm_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(vol), sigma=1.2)
            inv_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(1 - vol), sigma=1.2)
            return self._vectorized_soft(norm_transform + inv_transform, alpha, beta)
        else:
            return vol


    def _return_matted_transform_tensor(self, vol, alpha=20.0, beta=1.1, use_mask=True):
        if use_mask:
            #vol_blurred = ndimage.gaussian_filter(vol, sigma=0.2)
            norm_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(vol), sigma=1.2)
            inv_transform = ndimage.gaussian_filter(ndimage.distance_transform_edt(1 - vol), sigma=1.2)
            return self._vectorized_soft(norm_transform + inv_transform, alpha, beta)
        else:
            return vol




    def _gblur(self, vol, kern_size=(5,5,5), sigma=(1.0,1.0,1.0), multiplier=5):
        #### torch dilation to get boundary of superstructs
        dil_torch_kernel = th.ones(3,3,3)
        kernel_tensor = dil_torch_kernel[None, None, :] # size: (1, 1, 3, 3)
        dil_res = th.clamp(F.conv3d(vol, kernel_tensor, padding='same'),min=0,max=1)

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = th.meshgrid(
            [
                th.arange(size, dtype=th.float32)
                for size in kern_size
            ]
        )
        for size, std, mgrid in zip(kern_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1/(std*math.sqrt(2*math.pi)) * \
                      th.exp(-((mgrid - mean)/(2 * std))**2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / th.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        g_filtered_vol = F.conv3d(dil_res-vol,kernel,padding='same')*multiplier
        return g_filtered_vol





    def _smoothmax(self,vol_list,alpha=0.01,multiplier=1.0):

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
        return multiplier*vol_out


    def _smoothmax_tensor(self,vol_list,alpha=0.01,multiplier=1.0):
        return 0


    def forward(self, displacement):
        if self._relax_regularization:
            #warped_mask_matted_list = []
            local_weight_tensor = th.ones_like(self._mask_list[0].image)
            for i in range(0,len(self._mask_list)):
                # compute displacement field
                grid1 = T.utils.compute_grid(self._mask_list[i].size, dtype=self._mask_list[i].dtype, device=self._mask_list[i].device)
                displacement_upd = grid1 + displacement

                warped_mask = F.grid_sample(self._mask_list[i].image, displacement_upd)
                warped_blurred_mask = self._gblur(warped_mask,kern_size=(5,5,5), sigma=(1.5,1.5,1.5))
                local_weight_tensor += warped_blurred_mask
                #warped_mask_np = warped_mask.squeeze().detach().numpy()
                #mask_matted = self._return_matted_transform(warped_mask_np, alpha=20.0, beta=1.1, use_mask=self._using_mask)
                #warped_mask_matted_list.append(mask_matted)
            #save_nifti("/Users/markolchanyi/Desktop/Edlow_Brown/Projects/datasets/ex_vivo_test_data/EXC012/scratch/regularization_field_new.nii.gz",local_weight_tensor.detach().cpu().numpy()[0,0,...],np.eye(4))

            #smax_mask = self._smoothmax(warped_mask_matted_list,alpha=0.15,multiplier=1.0)
            #smask_matted_torch = th.from_numpy(smax_mask)
            if self._using_mask:
                #return self.return_loss(self._regulariser(displacement,smask_matted_torch))
                return self.return_loss(self._regulariser(displacement,local_weight_tensor))
            else:
                return 0
                #return self.return_loss(self._regulariser(displacement))
        else:
            return 0
            #return self.return_loss(self._regulariser(displacement,hold_mask=None))




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
