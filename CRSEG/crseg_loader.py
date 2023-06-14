import sys
import shutil
import os
import numpy as np
import nibabel as nib
import nibabel.processing
import matplotlib.pyplot as plt
import torch as th
from utils import print_no_newline
from skimage import morphology, filters
from scipy.ndimage import gaussian_filter
from dipy.io.image import load_nifti, save_nifti
from crseg_utils import resample,mean_threshold,crop_around_COM,\
    align_COM_masks,res_match_and_rescale,max_threshold,normalize_volume,\
    normalize_volume_mean_std,join_labels

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../airlab/airlab/")))) #add airlab dir to working dir

import airlab as al


################ MAIN LOADER FUNCTION ####################

def prepare_vols_multichan(test_path_list,
                       test_WM_mask_path,
                       atlas_path_list,
                       atlas_WM_mask_path,
                       savepath,
                       label_array,
                       save_affine=np.eye(4),
                       res_atlas=0.5,
                       res_test=0.75,
                       speed_crop=True,
                       pre_affine_step=True,
                       rotate_atlas=False,
                       tolerance=[50, 50],
                       loss_region_sigma=25.0):

    """
    This is the main function to load and tranform the multichannel atlas and target volumes,
    including the lists of target and atlas masks. Format for masks should be:

    [target mask1 path, target mask2 path, ...]
    [atlas mask1 path, atlas mask2 path, ...]


    :loss_region sigma is an optional variable for determine the volume of the region where NGF
    loss is calculated. Smaller values would account for a tigher loss region but run the risk of
    missing important fiducials

    """

    assert test_path_list[0].endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % test_path_list[0]
    assert atlas_path_list[0].endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % test_path_list[0]


    #### set glob params and import stuff ####
    # set the used data type
    ddtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")
    n_channels = len(test_path_list)
    test_list = []
    atlas_list = []
    test_masks_array_list = []
    atlas_masks_array_list = []

    print_no_newline("loading up all volumes for airlab...")

    for i in range(0,n_channels):
        test_foo = nib.load(test_path_list[i])
        atlas_foo = nib.load(atlas_path_list[i])
        test_header = test_foo.header
        atlas_header = atlas_foo.header

        # conform test -> atlas headers. for intestities use 3rd order splines
        #print_no_newline("conforming test volume " + str(i) + " ...")
        #test_foo = nib.processing.conform(test_foo,out_shape=atlas_header.get_data_shape(), voxel_size=atlas_header.get_zooms(), order=3)
        #print("done")

        test_foo_np = np.array(test_foo.dataobj)
        atlas_foo_np = np.array(atlas_foo.dataobj)
        np.nan_to_num(test_foo_np)
        np.nan_to_num(atlas_foo_np)
        #test_foo_np = normalize_volume_mean_std(test_foo_np)
        #atlas_foo_np = normalize_volume_mean_std(atlas_foo_np)

        test_list.append(normalize_volume(test_foo_np))
        atlas_list.append(normalize_volume(atlas_foo_np))


        del test_foo
        del atlas_foo
        del test_foo_np
        del atlas_foo_np


    ## WM masks are all in a single LUT volume, so load them into a similar list fashion
    test_wm_masks_vol = nib.load(test_WM_mask_path)
    atlas_wm_masks_vol = nib.load(atlas_WM_mask_path)
    test_wm_masks_header = test_wm_masks_vol.header
    atlas_wm_masks_header = atlas_wm_masks_vol.header

    affine = atlas_wm_masks_vol.affine

    # conform WM masks to atlas space, this should be nearest neighbor (0th order)
    #test_wm_masks_vol = nib.processing.conform(test_wm_masks_vol,out_shape=atlas_wm_masks_header.get_data_shape(), voxel_size=atlas_wm_masks_header.get_zooms(), order=0)
    test_wm_masks_vol = np.array(test_wm_masks_vol.dataobj)
    atlas_wm_masks_vol = np.array(atlas_wm_masks_vol.dataobj)


    for i in range(0,label_array.size):
        test_wm_masks_vol_copy = test_wm_masks_vol.copy()
        atlas_wm_masks_vol_copy = atlas_wm_masks_vol.copy()

        test_wm_masks_vol_copy[test_wm_masks_vol_copy != label_array[i]] = 0
        test_wm_masks_vol_copy[test_wm_masks_vol_copy == label_array[i]] = 1
        atlas_wm_masks_vol_copy[atlas_wm_masks_vol_copy != label_array[i]] = 0
        atlas_wm_masks_vol_copy[atlas_wm_masks_vol_copy == label_array[i]] = 1

        test_masks_array_list.append(test_wm_masks_vol_copy)
        atlas_masks_array_list.append(atlas_wm_masks_vol_copy)

        ### just for debugging of transformation model
        atlas_wm_save_vol = nib.Nifti1Image(atlas_wm_masks_vol_copy,affine=affine)
        nib.save(atlas_wm_save_vol,os.path.join("/Users/markolchanyi/Desktop/Edlow_Brown/Projects/testing/CRSEG_testing/subject_115320/CRSEG_outputs/wm_outputs","label_" + str(label_array[i]) + ".nii.gz"))

    del test_wm_masks_vol_copy
    del atlas_wm_masks_vol_copy

    if rotate_atlas is True:
        for i in range(0,n_channels):
            atlas_list[i] = np.rot90(atlas_list[i],k=3,axes=(1, 2))
        for i in range(0,label_array.size):
            atlas_masks_array_list[i] = np.rot90(atlas_masks_array_list[i],k=3,axes=(1, 2))
            atlas_masks_array_list[i] = max_threshold(atlas_masks_array_list[i])

        # strictly for finding COM
        joint_wm_test_mask = join_labels(test_wm_masks_vol)
        joint_wm_atlas_mask = join_labels(np.rot90(atlas_wm_masks_vol,k=3,axes=(1, 2)))


    else:
        atlas_masks_array_list[i] = max_threshold(atlas_masks_array_list[i])

        # strictly for finding COM
        joint_wm_test_mask = join_labels(test_wm_masks_vol)
        joint_wm_atlas_mask = join_labels(atlas_wm_masks_vol)

        del test_wm_masks_vol
        del atlas_wm_masks_vol






    """

    Preprocess all masks, including the option to crop around a chosen mask COM:

    if volumes are large, consider localizing the registration to
    the chosen mask COM and cropping out the entire periphary. This will GREATLY
    speed up computation time.

    """
    test_masks_list_reshaped = []
    atlas_masks_list_reshaped = []
    test_masks_list_reshaped_cropped_aligned = []
    atlas_masks_list_reshaped_cropped_aligned = []
    fixed_mask_list = []
    moving_mask_list = []

    counter = 0
    for i in range(0,label_array.size):
        if not pre_affine_step:
            atlas_mask1_reshaped, test_mask1_reshaped, mask_p_matrix = res_match_and_rescale(atlas_masks_array_list[i],
                                                                                        test_masks_array_list[i],
                                                                                        res_atlas,
                                                                                        res_test_base,
                                                                                        resample_factor,
                                                                                        res_flip=True)
            test_mask1_reshaped = max_threshold(test_mask1_reshaped)
            atlas_mask1_reshaped = max_threshold(atlas_mask1_reshaped)

            test_masks_list_reshaped.append(test_mask1_reshaped)
            atlas_masks_list_reshaped.append(atlas_mask1_reshaped)

            aligned_atlas_mask_reshaped,_ = align_COM_masks(test_mask1_reshaped,
                                                atlas_mask1_reshaped,
                                                test_masks_list_reshaped[0],
                                                atlas_masks_list_reshaped[0])

            test_masks_list_reshaped_cropped_aligned.append(test_mask1_reshaped)
            atlas_masks_list_reshaped_cropped_aligned.append(aligned_atlas_mask_reshaped)

        else:
            test_masks_list_reshaped_cropped_aligned.append(test_masks_array_list[i])
            atlas_masks_list_reshaped_cropped_aligned.append(atlas_masks_array_list[i])
            test_masks_list_reshaped.append(test_masks_array_list[i])
            atlas_masks_list_reshaped.append(atlas_masks_array_list[i])

        if speed_crop:
            test_mask1_reshaped_foo,atlas_mask1_reshaped,_,_,crop_matrix1,crop_matrix2 = crop_around_COM(test_masks_list_reshaped_cropped_aligned[i],
                                                                                                      atlas_masks_list_reshaped_cropped_aligned[i],
                                                                                                      joint_wm_test_mask,
                                                                                                      joint_wm_atlas_mask,
                                                                                                      tolerance)

        test_mask1_reshaped = test_mask1_reshaped_foo
        ## binarize and clean masks
        #test_mask1_reshaped = max_threshold(test_mask1_reshaped)
        #where_objs = morphology.remove_small_objects(test_mask1_reshaped.astype(np.bool_), 5)
        #test_mask1_reshaped[where_objs < 0.5] = 0
        gauss_sigma = 0.25

        #test_mask1_reshaped = gaussian_filter(test_mask1_reshaped, sigma=gauss_sigma)
        #atlas_mask1_reshaped = gaussian_filter(atlas_mask1_reshaped, sigma=gauss_sigma)

        #test_mask1_reshaped = max_threshold(test_mask1_reshaped)
        #atlas_mask1_reshaped = max_threshold(atlas_mask1_reshaped)


        [s0, s1, s2] = test_mask1_reshaped.shape

        ### the loss region will always be the first mask pair in the list!
        if counter == 0:
            #fixed_loss_region = gaussian_filter(test_mask1_reshaped, sigma=loss_region_sigma)
            fixed_loss_region = np.ones_like(test_mask1_reshaped)
            moving_loss_region = np.ones_like(atlas_mask1_reshaped)
            #moving_loss_region = gaussian_filter(atlas_mask1_reshaped, sigma=loss_region_sigma)
            fixed_loss_region[fixed_loss_region > 0] = 1
            moving_loss_region[moving_loss_region > 0] = 1

            fixed_loss_region = al.Image(fixed_loss_region.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)
            moving_loss_region = al.Image(moving_loss_region.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)

        fixed_mask = al.Image(test_mask1_reshaped.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)
        moving_mask = al.Image(atlas_mask1_reshaped.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)
        fixed_mask_list.append(fixed_mask)
        moving_mask_list.append(moving_mask)

        counter += 1



    fixed_image_list = []
    moving_image_list = []
    for i in range(0,n_channels):
        if not pre_affine_step:
            atlas_reshaped, test_reshaped, test_p_matrix = res_match_and_rescale(atlas_list[i],
                                                                                test_list[i],
                                                                                res_atlas,
                                                                                res_test_base,
                                                                                resample_factor,
                                                                                res_flip=True)

            atlas_reshaped_aligned,_ = align_COM_masks(test_reshaped,
                                                    atlas_reshaped,
                                                    test_masks_list_reshaped[0],
                                                    atlas_masks_list_reshaped[0])

        else:
            test_reshaped = test_list[i]
            atlas_reshaped_aligned = atlas_list[i]
            test_p_matrix = None


        if speed_crop:
            test_reshaped_cropped_foo,atlas_reshaped_cropped_foo,_,_,crop_matrix1,crop_matrix2 = crop_around_COM(test_reshaped,
                                                                                                        atlas_reshaped_aligned,
                                                                                                        joint_wm_test_mask,
                                                                                                        joint_wm_atlas_mask,
                                                                                                        tolerance)

        test_reshaped_cropped = test_reshaped_cropped_foo #gaussian_filter(test_reshaped_cropped_foo,sigma=0.1)


        gauss_sigma = 0.15
        test_reshaped_cropped = gaussian_filter(test_reshaped_cropped, sigma=gauss_sigma)
        #atlas_reshaped_cropped_foo = gaussian_filter(atlas_reshaped_cropped_foo, sigma=gauss_sigma)
        fixed_image_list.append(al.Image(test_reshaped_cropped.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype))
        moving_image_list.append(al.Image(atlas_reshaped_cropped_foo.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype))

    print(" done")



    return fixed_image_list,moving_image_list,fixed_mask_list,moving_mask_list,fixed_loss_region,moving_loss_region,test_p_matrix
