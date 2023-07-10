import sys
import shutil
import os
import time
import math
import numpy as np
import torch as th
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift
from scipy import ndimage, misc
from skimage import morphology, filters
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy import stats
from scipy import ndimage
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import misc
from scipy.spatial import distance
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
#matplotlib inline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../airlab/airlab/")))) #add airlab dir to working dir

import airlab as al



"""

This script holds muiltiple support functions for data loading and preprocessing
as well as test functions for ad hoc stuff..

"""


def stack_channels(vol,chan,ax,ddtype=np.float32):
    return np.stack((vol,chan.astype(np.float32)),axis=ax)




def rescale_and_pad_to_vol(vol,temp_vol):
    ds_factor = np.min([temp_vol.shape[0]/vol.shape[0],temp_vol.shape[1]/vol.shape[1],temp_vol.shape[2]/vol.shape[2]])
    vol_rescaled = rescale(vol, ds_factor, anti_aliasing=True)

    pad_s1 = int(math.floor((temp_vol.shape[0]-vol_rescaled.shape[0])/2))
    pad_s2 = int(math.floor((temp_vol.shape[1]-vol_rescaled.shape[1])/2))
    pad_s3 = int(math.floor((temp_vol.shape[2]-vol_rescaled.shape[2])/2))

    repad_s1 = temp_vol.shape[0] - pad_s1 - vol_rescaled.shape[0]
    repad_s2 = temp_vol.shape[1] - pad_s2 - vol_rescaled.shape[1]
    repad_s3 = temp_vol.shape[2] - pad_s3 - vol_rescaled.shape[2]

    out = np.pad(vol_rescaled,((pad_s1,repad_s1),(pad_s2,repad_s2),(pad_s3,repad_s3)),'constant')
    assert out.shape == temp_vol.shape, "resizing and padding did not produce similar dimensions!"
    return out




def get_jacobian_determinant(disp_field):

    disp_np = disp_field.detach().cpu().numpy()

    grad_ux = np.gradient(disp_np[...,0],axis=0)
    grad_uy = np.gradient(disp_np[...,0],axis=1)
    grad_uz = np.gradient(disp_np[...,0],axis=2)

    grad_vx = np.gradient(disp_np[...,1],axis=0)
    grad_vy = np.gradient(disp_np[...,1],axis=1)
    grad_vz = np.gradient(disp_np[...,1],axis=2)

    grad_wx = np.gradient(disp_np[...,2],axis=0)
    grad_wy = np.gradient(disp_np[...,2],axis=1)
    grad_wz = np.gradient(disp_np[...,2],axis=2)

    jac_det_volume = np.zeros_like(grad_ux)

    for i in range(0,jac_det_volume.shape[0]):
        for j in range(0,jac_det_volume.shape[1]):
            for k in range(0,jac_det_volume.shape[2]):
                jac_vox_mat = np.array([grad_ux[i,j,k],grad_uy[i,j,k],grad_uz[i,j,k],
                               grad_vx[i,j,k],grad_vy[i,j,k],grad_vz[i,j,k],
                               grad_wx[i,j,k],grad_wy[i,j,k],grad_wz[i,j,k]]).reshape(3,3)
                jac_det_volume[i,j,k] = scipy.linalg.det(jac_vox_mat)

    return jac_det_volume




def COM(volume,thresh,round_to_int=True):
    if round_to_int:
        return np.rint([np.average(indices).astype(int) for indices in np.where(volume > thresh)])
    else:
        return [np.average(indices).astype(int) for indices in np.where(volume > thresh)]




def mean_threshold(vol):
    mn = np.mean(vol)
    vol[vol < mn] = 0
    vol[vol > 0] = 1
    return vol



def max_threshold(vol):
    vol_out = np.zeros_like(vol).astype(np.float32)
    max = np.max(vol)
    vol_out[vol >= max/2] = 1
    return vol_out



def add_gauss_noise_to_binary_volume(vol,mean=0.0,std=1.0):
    noise_vol = vol + np.random.normal(mean, std, vol.shape)
    return binary_fill_holes(threshold(vol*noise_vol))




def res_match_and_rescale(vol,temp_vol_foo,res,res_temp,resample_factor=None, res_flip=True):
    if resample_factor is not None:
        res_temp *= resample_factor
        temp_vol_foo = rescale(temp_vol_foo, 1/resample_factor, anti_aliasing=True)

    if res_flip:
        res_match = res_temp/res
        vol_rescaled = vol
        temp_vol = rescale(temp_vol_foo, res_match, anti_aliasing=True)

    else:
        res_match = res/res_temp
        vol_rescaled = rescale(vol, res_match, anti_aliasing=True)
        temp_vol = temp_vol_foo

    # perform common padding to match shapes
    comp_index = [vol_rescaled.shape[i] < temp_vol.shape[i] for i in range(0,vol.ndim)]
    pad_matrix = np.zeros((3,2))
    pad_matrix_resflipped = np.zeros((3,2))
    pad_index_array = np.zeros(3)
    for j in range(0,vol.ndim):
        if comp_index[j]:
            pad_s = int(math.floor((temp_vol.shape[j]-vol_rescaled.shape[j])/2))
            repad_s = temp_vol.shape[j] - pad_s - vol_rescaled.shape[j]
            # gross but make explicit for now since np.pad doesn't have an axis setting...
            if j == 0:
                vol_rescaled = np.pad(vol_rescaled,((pad_s,repad_s),(0,0),(0,0)),'constant')
            if j == 1:
                vol_rescaled = np.pad(vol_rescaled,((0,0),(pad_s,repad_s),(0,0)),'constant')
            if j == 2:
                vol_rescaled = np.pad(vol_rescaled,((0,0),(0,0),(pad_s,repad_s)),'constant')
            pad_matrix_resflipped[j,:] = [pad_s,repad_s]
        else:
            pad_s = -int(math.floor((temp_vol.shape[j]-vol_rescaled.shape[j])/2))
            repad_s = vol_rescaled.shape[j] - pad_s - temp_vol.shape[j]
            # gross but make explicit for now since np.pad doesn't have an axis setting...
            if j == 0:
                temp_vol = np.pad(temp_vol,((pad_s,repad_s),(0,0),(0,0)),'constant')
            if j == 1:
                temp_vol = np.pad(temp_vol,((0,0),(pad_s,repad_s),(0,0)),'constant')
            if j == 2:
                temp_vol = np.pad(temp_vol,((0,0),(0,0),(pad_s,repad_s)),'constant')
            pad_matrix[j,:] = [pad_s,repad_s]

    if not res_flip:
        return vol_rescaled, temp_vol, pad_matrix
    else:
        return vol_rescaled, temp_vol, pad_matrix



def unpad(vol, pad_matrix):
    pw0 = tuple(pad_matrix[0,:].astype(int))
    pw1 = tuple(pad_matrix[1,:].astype(int))
    pw2 = tuple(pad_matrix[2,:].astype(int))

    pad_width = (pw0,pw1,pw2)
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return vol[tuple(slices)]



def vectorized_soft(m,alpha=4.0,beta=1.0):
    return alpha/(alpha*(1 + np.exp(-(alpha*(m-beta)))))




def return_matted_transform(vol,alpha,beta,sig=0.2,soft=True):
    vol_blurred = gaussian_filter(vol, sigma=sig)
    norm_transform = gaussian_filter(ndimage.distance_transform_edt(vol), sigma=sig)
    inv_transform = gaussian_filter(ndimage.distance_transform_edt(1 - vol), sigma=sig)
    if soft is True:
        return vectorized_soft(norm_transform + inv_transform,alpha,beta)
    else:
        return norm_transform + inv_transform




def smoothmax(vol1,vol2,alpha):
    vol_out = np.zeros_like(vol1)
    for i in range(0,vol1.shape[0]):
        for j in range(0,vol1.shape[1]):
            for k in range(0,vol1.shape[2]):
                p1 = vol1[i,j,k]
                p2 = vol2[i,j,k]
                vol_out[i,j,k] = (p1*math.exp(alpha*p1) + p2*math.exp(alpha*p2))/(math.exp(alpha*p1) + math.exp(alpha*p2))
    return vol_out





def displace_back(vol,cm_disp): #shift a joint domain volume back into original coords
    [r1, r2, r3] = cm_disp
    if isinstance(vol, np.ndarray):
        out = np.roll(vol, (-r1,-r2,-r3), axis=(0,1,2))
    else:
        out = np.roll(vol.numpy(), (-r1,-r2,-r3), axis=(0,1,2))
    return out




def shift_vol(vol,shift,ax=0):
    return np.roll(vol, -shift, axis=ax)




def crop_around_borders(vol,mag,ax=0):
    if ax==0:
        return vol[mag-1:-mag,:,:]
    elif ax==1:
        return vol[:,mag-1:-mag,:]
    else:
        return vol[:,:,mag-1:-mag]



def join_labels(vol):
    vol[vol > 0] = 1
    return vol




def crop_around_COM(vol1,vol2,brainstem_mask1,brainstem_mask2,tolerance):
    shape1 = vol1.shape
    shape2 = vol2.shape
    COM1 = COM(brainstem_mask1,0.5)
    COM2 = COM(brainstem_mask2,0.5)

    ###### add explicitly only if there is an affine step in order to better center the brainstem
    #COM1[1] += 15
    #COM2[1] += 15
    #COM1[2] -= 10
    #COM2[2] -= 10

    bound1 = np.zeros((3,2))
    bound2 = np.zeros((3,2))
    for i in range(0,3):
        bound1[i,0] = max(0,COM1[i] - tolerance[0])
        bound1[i,1] = max(0,shape1[i] - COM1[i] - tolerance[0])
        bound2[i,0] = max(0,COM2[i] - tolerance[1])
        bound2[i,1] = max(0,shape2[i] - COM2[i] - tolerance[1])

    vol1_cropped = unpad(vol1,bound1)
    brainstem_mask1_cropped = unpad(brainstem_mask1,bound1)
    vol2_cropped = unpad(vol2,bound2)
    brainstem_mask2_cropped = unpad(brainstem_mask2,bound2)

    return vol1_cropped,vol2_cropped,brainstem_mask1_cropped,brainstem_mask2_cropped,bound1,bound2




def uncrop_volume(vol,crop_matrix):
    crop_matrix = crop_matrix.astype(int)
    out = np.pad(vol,((crop_matrix[0,0],crop_matrix[0,1]),(crop_matrix[1,0],crop_matrix[1,1]),(crop_matrix[2,0],crop_matrix[2,1])),'constant')
    return out




def move_to_image_center(vol,thresh=0.1):
    com = COM(vol,thresh)
    imag_cent = COM(np.ones_like(vol),thresh)
    r1 = com[0] - imag_cent[0]
    r2 = com[1] - imag_cent[1]
    r3 = com[2] - imag_cent[2]
    return np.roll(vol, (-r1,-r2,-r3), axis=(0,1,2))



def normalize_volume(vol):
    vol_windowed = vol
    return (vol_windowed - np.min(vol_windowed)) / (np.max(vol_windowed) - np.min(vol_windowed))



def normalize_volume_mean_std(vol,factor=1):
    vol = vol - vol.mean()
    vol = vol/(factor*vol.std())

    vol += 0.5
    #clip
    vol[vol < 0] = 0
    vol[vol > 1] = 1

    return vol



def resample(vol,orig_res,new_res):
    rs_factor = new_res/orig_res
    temp_ds = rescale(vol, 1/rs_factor, anti_aliasing=False)
    temp_us = resize(temp_ds, (vol.shape[0],vol.shape[1],vol.shape[2]), anti_aliasing=False)

    if np.amax(temp_us) < 0.5:
        temp_us *= 1e10

    return temp_us




def align_COM_masks(fvol,mvol,fmask,mmask): # will only roll vol2
    assert fvol.shape == mvol.shape and fmask.shape == mmask.shape, "volumes have different dimensions!"
    thresh = 0.5
    mCOM = COM(mmask,thresh,round_to_int=False)
    fCOM = COM(fmask,thresh,round_to_int=False)

    dist = np.array(fCOM) - np.array(mCOM)
    m0, m1, m2 = dist

    shifted_mvol = np.roll(mvol, (m0, m1, m2) ,axis=(0,1,2))
    shifted_mmask = np.roll(mmask, (m0, m1, m2) ,axis=(0,1,2))
    return shifted_mvol, shifted_mmask





def construct_label_volume(inpdir,outname,ignore=None):
    counter = 1
    for subdir, dirs, files in os.walk(inpdir):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if filename == ignore:
                continue
            else:
                if counter == 1:
                    vol_tmp, affine_v = load_nifti(os.path.join(subdir, file), return_img=False)
                    vol_empty = np.zeros_like(vol_tmp)

                vol, affine_v = load_nifti(os.path.join(subdir, file), return_img=False)
                vol_empty[vol > 0] = int(filename)
                counter += 1

    save_nifti(subdir + outname,vol_empty,affine_v)





def template_volume_to_labels(template_path,output_dir,num_labels=80):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vol_in,dummy_affine = load_nifti(template_path, return_img=False)

    for i in range(1,num_labels):
        vol_out = np.zeros_like(vol_in)

        if i in vol_in:
            vol_out[vol_in == i] = 1
            save_nifti(output_dir + str(i) + ".nii",vol_out,dummy_affine)



def affine_trans(atlas_path_list,test_path_list,test_mask_path,scratch_dir):
    print("--------------- RUNNING AFFINE STEP -----------------------")
    os.system("reg_aladin -ref " + atlas_path_list[1] + " -flo " + test_path_list[1] + " -res " + os.path.join(scratch_dir,"b0_affine_transformed.nii.gz") + " -aff " + os.path.join(scratch_dir,"affine_mat.txt"))
    os.system("reg_resample -ref " + atlas_path_list[1] + " -flo " + test_path_list[0] + " -res " + os.path.join(scratch_dir,"fa_affine_transformed.nii.gz") + " -trans " + os.path.join(scratch_dir,"affine_mat.txt") + " -inter 2")
    os.system("reg_resample -ref " + atlas_path_list[1] + " -flo " + test_path_list[1] + " -res " + os.path.join(scratch_dir,"b0_affine_transformed.nii.gz") + " -trans " + os.path.join(scratch_dir,"affine_mat.txt") + " -inter 2")

    ### make sure nearest neighbor interpolation is used for label volume
    os.system("reg_resample -ref " + atlas_path_list[1] + " -flo " + test_mask_path + " -res " + os.path.join(scratch_dir,"WM_masks_affine_transformed.nii.gz") + " -trans " + os.path.join(scratch_dir,"affine_mat.txt") + " -inter 0")

    ### resample everything to the same space
    os.system("mri_convert " + os.path.join(scratch_dir,"WM_masks_affine_transformed.nii.gz") + " " os.path.join(scratch_dir,"WM_masks_affine_transformed.nii.gz") + " -rl " + atlas_path_list[1] + " -odt float")
    os.system("mri_convert " + os.path.join(scratch_dir,"b0_affine_transformed.nii.gz") + " " os.path.join(scratch_dir,"b0_affine_transformed.nii.gz") + " -rl " + atlas_path_list[1] + " -odt float")
    os.system("mri_convert " + os.path.join(scratch_dir,"fa_affine_transformed.nii.gz") + " " os.path.join(scratch_dir,"fa_affine_transformed.nii.gz") + " -rl " + atlas_path_list[1] + " -odt float")


    ### save inverse affine for propagation step
    os.system("reg_transform -invAff " + os.path.join(scratch_dir,"affine_mat.txt") + " " +  os.path.join(scratch_dir,"inverse_affine_mat.txt"))





class ImageSliceViewer3D:
    def __init__(self, volume, figsize=(8,8), cmap='plasma'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]

        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y',
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice,
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False,
            description='Image Slice:'))

    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap),
            vmin=self.v[0], vmax=self.v[1])
