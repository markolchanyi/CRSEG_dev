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
from torch.autograd import gradcheck
import airlab as al
import os

import numpy as np
import pickle

from .. import transformation as T
from ..transformation import utils as tu
from ..utils import kernelFunction as utils

from packaging import version
from scipy import ndimage

import sys
sys.path.insert(1, '/autofs/vast/nicc_003/users/olchanyi/CRSEG_dev')  # REMEMBER THIS IS THE DEV VERSION
import varifold as vf
import distance_functions as df
from dipy.io.image import load_nifti, save_nifti
from torchmcubes import marching_cubes, grid_interp




# Loss base class (standard from PyTorch)
class _PairwiseImageLoss_compound(th.nn.modules.Module):
    def __init__(self,
    casepath,
    fixed_image_list,
    moving_image_list,
    fixed_loss_region,
    fixed_mask_list,
    moving_loss_region,
    moving_mask_list,
    loss_region_weight,
    mask_weights,
    channel_weight,
    epsilon,
    size_average,
    reduce,
    single_channel,
    no_superstructs,
    varifold,
    vf_sigma,
    step_size,
    generate_mesh,
    fixed_norms_list,
    fixed_cts_list,
    moving_verts_list,
    moving_faces_list,
    Dice,
    MSE):
        super(_PairwiseImageLoss_compound, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._name = "parent"

        self._warped_moving_list = None
        self._warped_moving_mask_list = None
        self._weight = 1
        self._single_channel = single_channel
        self._no_superstructs = no_superstructs
        self._use_varifold = varifold
        self._vf_sigma = vf_sigma
        self._step_size = step_size
        self._use_mse = MSE
        self._use_dice = Dice
        self._casepath = casepath

        self._fixed_image_list = fixed_image_list
        self._moving_image_list = moving_image_list

        self._fixed_loss_region = fixed_loss_region
        self._moving_loss_region = moving_loss_region
        self._fixed_mask_list = fixed_mask_list
        self._moving_mask_list = moving_mask_list

        self._loss_region_weight = loss_region_weight
        self._mask_weights = mask_weights
        self._channel_weight = channel_weight

        self._grid = None

        self._generate_mesh = generate_mesh
        self._fixed_norms_list=fixed_norms_list
        self._fixed_cts_list=fixed_cts_list
        self._moving_verts_list=moving_verts_list
        self._moving_faces_list=moving_faces_list


        self._grid = T.utils.compute_grid(self._moving_image_list[0].size, dtype=self._moving_image_list[0].dtype,
        device=self._moving_image_list[0].device)

        self._dtype = self._moving_image_list[0].dtype
        self._device = self._moving_image_list[0].device

    @property
    def name(self):
        return self._name


    def GetCurrentMask(self, displacement, moving, fixed):
        """
        Computes a mask defining if pixels are warped outside the image domain, or if they fall into
        a fixed image mask or a warped moving image mask.
        return (Tensor): maks array
        """
        # exclude points which are transformed outside the image domain
        mask = th.zeros_like(self._fixed_image_list[0].image, dtype=th.uint8, device=self._device)
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
    casepath,
    fixed_image_list,
    moving_image_list,
    fixed_loss_region,
    fixed_mask_list,
    moving_loss_region,
    moving_mask_list,
    loss_region_weight=1.0,
    mask_weights=None,
    channel_weight=1.0,
    epsilon=1e-5,
    size_average=True,
    reduce=True,
    single_channel=False,
    no_superstructs=False,
    varifold=True,
    vf_sigma=30.0,
    step_size=2.0,
    generate_mesh=False,
    fixed_norms_list=None,
    fixed_cts_list=None,
    moving_verts_list=None,
    moving_faces_list=None,
    Dice=False,
    MSE=False):
        super(COMPOUND, self).__init__(casepath,
        fixed_image_list,
        moving_image_list,
        fixed_loss_region,
        fixed_mask_list,
        moving_loss_region,
        moving_mask_list,
        loss_region_weight,
        mask_weights,
        channel_weight,
        epsilon,
        size_average,
        reduce,
        single_channel,
        no_superstructs,
        varifold,
        vf_sigma,
        step_size,
        generate_mesh,
        fixed_norms_list,
        fixed_cts_list,
        moving_verts_list,
        moving_faces_list,
        Dice,
        MSE)

        self._name = "compound_varifold_loss"

        self._dim = fixed_image_list[0].ndim
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
            dx_list = []
            dy_list = []
            dz_list = []
            norm_list = []
            self._epsilon_list = []
            self._ng_fixed_image_list = []

            for i in range(0,len(fixed_image_list)):
                dx_list.append((fixed_image_list[i].image[..., 1:, 1:, 1:] - fixed_image_list[i].image[..., :-1, 1:, 1:]) * fixed_image_list[i].spacing[0])
                dy_list.append((fixed_image_list[i].image[..., 1:, 1:, 1:] - fixed_image_list[i].image[..., 1:, :-1, 1:]) * fixed_image_list[i].spacing[1])
                dz_list.append((fixed_image_list[i].image[..., 1:, 1:, 1:] - fixed_image_list[i].image[..., 1:, 1:, :-1]) * fixed_image_list[i].spacing[2])

                if self._epsilon is None:
                    with th.no_grad():
                        self._epsilon_list.append(th.mean(th.abs(dx_list[i]) + th.abs(dy_list[i]) + th.abs(dz_list[i]))*0.005)

                norm_list.append(th.sqrt(dx_list[i].pow(2) + dy_list[i].pow(2) + dz_list[i].pow(2) + self._epsilon_list[i] ** 2))

                self._ng_fixed_image_list.append(F.pad(th.cat((dx_list[i], dy_list[i], dz_list[i]), dim=1) / norm_list[i], (0, 1, 0, 1, 0, 1)))

                ###########

                self._ngf_loss = self._ngf_loss_3d





    def _retrieve_np_displacement(self, displacement):
        return displacement.detach().cpu().numpy()




    def _mean_threshold(self, vol):
        mn = np.mean(vol)
        vol[vol < mn] = 0
        vol[vol > 0] = 1
        return vol




    def _calculate_and_save_hausdorff(self, vol_list1, vol_list2, filepath):
        savelist = []
        mask_hdd_arr = np.zeros(len(vol_list1))
        for i in range(0,len(vol_list1)):
            vol1 = vol_list1[i].detach().numpy()
            vol1 = vol1[0,0,:,:,:]
            vol2 = vol_list2[i].numpy()
            vol2 = vol2[...,:,:,:]
            mask_hdd_arr[i] = df.scipy_hd95(vol1,vol2, voxelspacing=0.5, connectivity=8)

        savelist.append(mask_hdd_arr)

        if not os.path.exists(filepath):
            with open(filepath,'wb') as wfp:
                pickle.dump(savelist,wfp)
        else:
            with open(filepath,'rb') as rfp:
                scores = pickle.load(rfp)

            scores.append(savelist)

            with open(filepath,'wb') as wfp:
                pickle.dump(scores,wfp)




    def _rbf_kernel_torch(self, inp, sigma=7):

        #print("INP: ",inp)

        stacked_in = inp.unsqueeze(0).expand((inp.size(0),inp.size(0),3))
        diff = (stacked_in - th.transpose(stacked_in,0,1)).pow(2).sum(-1)/(2*(sigma**2))


        #print("RBF: ", th.exp(-diff))

        return th.exp(-diff)




    def _varifold_distance_torch(self,norm1,norm2,ct1,ct2):

            cts = th.vstack((ct1,ct2))
            norms = th.vstack((norm1,norm2))

            hilbert_mat = self._rbf_kernel_torch(cts)*(th.matmul(norms,th.transpose(norms,0,1)).pow(2))

            hilbert_distance = th.sum(hilbert_mat[0:ct1.size(0),0:ct1.size(0)])
            hilbert_distance += th.sum(hilbert_mat[ct1.size(0):,ct1.size(0):])
            hilbert_distance -= 2*th.sum(hilbert_mat[ct1.size(0):,0:ct1.size(0)])

            return hilbert_distance




    def _ngf_loss_2d(self, warped_image):
        dx = (warped_image[..., 1:, 1:] - warped_image[..., :-1, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:] - warped_image[..., 1:, :-1]) * self._moving_image.spacing[1]
        norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))




    def _ngf_loss_3d(self, warped_image):

        dx = (warped_image[..., 1:, 1:, 1:] - warped_image[..., :-1, 1:, 1:]) * self._moving_image_list[0].spacing[0]
        dy = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, :-1, 1:]) * self._moving_image_list[0].spacing[1]
        dz = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, 1:, :-1]) * self._moving_image_list[0].spacing[2]

        if self._epsilon is None:
            with th.no_grad():
                self._epsilon = th.mean(th.abs(dx) + th.abs(dy) + th.abs(dz)) * 0.005

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))


    def _compound_loss_3d(self, displacement):

        """

        keep EVERYTHING in this memory function a torch tensor operation!!!!!
        reason: gradient recovery cannot happen during adam or LBFGS if numpy
        conversions are performed

        """

        vf_loss = 0
        mse = 0
        for i in range(0,len(self._moving_mask_list)):
            #### try explicit re-meshing
            moving_warped_mask = F.grid_sample(self._moving_mask_list[i].image,displacement)
            fixed_mask = self._fixed_mask_list[i].image


            """
            verts_moving, faces_moving = marching_cubes(moving_warped_mask[0,0,...], 0.4)
            verts_fixed, faces_fixed = marching_cubes(fixed_mask[0,0,...], 0.4)

            cts_normals_moving = th.zeros_like(verts_moving)
            cts_moving = th.zeros_like(verts_moving)

            cts_normals_fixed = th.zeros_like(verts_fixed)
            cts_fixed = th.zeros_like(verts_fixed)


            print("size is: ",verts_moving.size(dim=0))
            for j in range(0,verts_moving.size(dim=0)):
                norm_vec = th.cross((verts_moving[faces_moving.long()][j,1,:]-verts_moving[faces_moving.long()][j,0,:]),(verts_moving[faces_moving.long()][j,2,:]-verts_moving[faces_moving.long()][j,0,:]))
                nrm = th.linalg.norm(norm_vec)
                if nrm != 0:
                    norm_vec /= nrm
                cts_normals_moving[j,:] = norm_vec
                cts_moving[j,:] = (verts_moving[faces_moving.long()][j,0,:] + verts_moving[faces_moving.long()][j,1,:] + verts_moving[faces_moving.long()][j,2,:])/3


            for j in range(0,verts_fixed.size(dim=0)):
                norm_vec = th.cross((verts_fixed[faces_fixed.long()][j,1,:]-verts_fixed[faces_fixed.long()][j,0,:]),(verts_fixed[faces_fixed.long()][j,2,:]-verts_fixed[faces_fixed.long()][j,0,:]))
                nrm = th.linalg.norm(norm_vec)
                if nrm != 0:
                    norm_vec /= nrm
                cts_normals_fixed[j,:] = norm_vec
                cts_fixed[j,:] = (verts_fixed[faces_fixed.long()][j,0,:] + verts_fixed[faces_fixed.long()][j,1,:] + verts_fixed[faces_fixed.long()][j,2,:])/3
            """


            #mse += self._varifold_distance_torch(cts_normals_fixed,cts_normals_moving,cts_fixed,cts_moving)

            mse += (self._mask_weights[i]*(moving_warped_mask - fixed_mask).pow(2)).mean()
            #save_nifti(self._casepath + "scratch/warped_mask_new" + str(i) + ".nii.gz",moving_warped_mask.detach().cpu().numpy()[0,0,...],np.eye(4))
            #save_nifti(self._casepath + "scratch/fixed_mask_new" + str(i) + ".nii.gz",self._fixed_mask_list[i].image.detach().cpu().numpy()[0,0,...],np.eye(4))
            #save_nifti(self._casepath + "scratch/fixed_im_new" + str(i) + ".nii.gz",self._fixed_image_list[0].image.detach().cpu().numpy()[0,0,...],np.eye(4))
            #print("MSE loss for mask" + str(i) + " : ", mse)
            #moving_warped_mask = moving_warped_mask.detach().cpu().numpy()[0,0,...]
            #fixed_mask = self._fixed_mask_list[i].image.detach().cpu().numpy()[0,0,...]
            #_,_,self._moving_cts,self._moving_norms = vf.mesh(self._mean_threshold(moving_warped_mask),step_size=self._step_size)
            #vf_loss += self._mask_weights[i]*vf.varifold_distance_FAST(self._moving_norms, self._fixed_norms_list[i], self._moving_cts, self._fixed_cts_list[i], sigma=self._vf_sigma)

        #val_total = val_ngf + vf_loss
        return mse


    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement
        self._n_channels = len(self._fixed_image_list)

        # compute current mask and dummy ones mask
        self._fixed_loss_region = self._fixed_loss_region.to(dtype=self._dtype, device=self._device)
        self._loss_region = super(COMPOUND, self).GetCurrentMask(displacement, self._moving_loss_region, self._fixed_loss_region)

        self._warped_moving_list = []
        self._ng_warped_image_list = []


        ### warp and calculate normalized gradients
        value = 0
        for i in range(0,self._n_channels):
            self._warped_moving_list.append(F.grid_sample(self._moving_image_list[i].image, displacement))
            self._ng_warped_image_list.append(self._ngf_loss(self._warped_moving_list[i]))
            for dim in range(self._dim):
                value += self._channel_weight * self._ng_warped_image_list[i][:, dim, ...] * self._ng_fixed_image_list[i][:, dim, ...]
        val_ngf = 0.5 * th.masked_select(-value.pow(2), self._loss_region).mean()



        # @TODO update to comply with mask list
        ### VARIFOLD MASK LOSS
        if self._use_varifold and not self._use_mse and not self._use_dice:
            #disp_np = self._retrieve_np_displacement(displacement)
            #vf_loss = 0
            #for i in range(0,len(self._moving_mask_list)):
            #    #self._moving_cts, self._moving_norms = vf.update_mesh(self._moving_verts_list[i],self._moving_faces_list[i],disp_np)
            #    #### try explicit re-meshing
            #    moving_warped_mask = F.grid_sample(self._moving_mask_list[i].image,displacement)
            #    moving_warped_mask = moving_warped_mask.detach().cpu().numpy()[0,0,...]
            #    fixed_mask = self._fixed_mask_list[i].image.detach().cpu().numpy()[0,0,...]
            #    save_nifti("/Users/markolchanyi/Desktop/Edlow_Brown/Projects/datasets/ex_vivo_test_data/EXC007/scratch/warped_mask" + str(i) + ".nii.gz",moving_warped_mask,np.eye(4))
            #    save_nifti("/Users/markolchanyi/Desktop/Edlow_Brown/Projects/datasets/ex_vivo_test_data/EXC007/scratch/fixed_mask" + str(i) + ".nii.gz",fixed_mask,np.eye(4))
            #    _,_,self._moving_cts,self._moving_norms = vf.mesh(self._mean_threshold(moving_warped_mask),step_size=self._step_size)
            #    vf_loss += self._mask_weights[i]*vf.varifold_distance_FAST(self._moving_norms, self._fixed_norms_list[i], self._moving_cts, self._fixed_cts_list[i], sigma=self._vf_sigma)
            #print("VF loss for mask" + str(i) + " : ", vf_loss)

            vf_loss = self._compound_loss_3d(displacement)
            #print("starting gradcheck")
            #test = gradcheck(self._ngf_loss, displacement, eps=1e-6, atol=1e-4)
            #print("is grad satisfied?: ", test)
            #print("VF loss for mask" + str(i) + " : ", vf_loss)

            val_total = vf_loss + val_ngf
            print("split -- VARIFOLD: ", vf_loss, " NGF: ", val_ngf)



        ### MEAN SQUARE ERROR MASK LOSS
        elif self._use_mse and not self._use_varifold and not self._use_dice:

            ### splines warp the mask list
            self._moving_mask_warped_list = []
            for i in range(0,len(self._moving_mask_list)):
                self._moving_mask_warped_list.append(F.grid_sample(self._moving_mask_list[i].image,displacement))
                self._fixed_mask_list[i] = self._fixed_mask_list[i].to(dtype=self._dtype, device=self._device)

            for i in range(0,len(self._moving_mask_list)):
                if i==0:
                    val_MSE = self._mask_weights[i]*(self._moving_mask_warped_list[0] - self._fixed_mask_list[0].image).pow(2)
                else:
                    val_MSE += self._mask_weights[i]*(self._moving_mask_warped_list[i] - self._fixed_mask_list[i].image).pow(2)
            val_MSE = th.masked_select(val_MSE,self._loss_region).mean()

            val_total = val_ngf + val_MSE
            print("split -- MSE: ", val_MSE, " NGF: ", val_ngf)

        ### DICE MASK LOSS
        elif self._use_dice and not self._use_mse and not self._use_varifold:
            val_dice = self._mask1_weight*vf.dice_loss(self._moving_mask1_warped.detach().numpy(),self._fixed_mask1.image.detach().numpy(),prethresh=True) + self._mask2_weight*vf.dice_loss(self._moving_mask2_warped.detach().numpy(),self._fixed_mask2.image.detach().numpy(),prethresh=True)
            val_total = val_ngf + val_dice
            print("split -- Dice: ", val_dice, " NGF: ", val_ngf)

        else:
            raise Exception('pick one of the loss functions: varifold, MSE, Dice!')
            sys.exit('none or multiple loss funcs picked')


        return val_total
