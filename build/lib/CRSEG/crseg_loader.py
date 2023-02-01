import sys
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from skimage import morphology, filters
from scipy.ndimage import gaussian_filter
from dipy.io.image import load_nifti, save_nifti
from crseg_utils import resample,mean_threshold,crop_around_COM,\
    align_COM_masks,res_match_and_rescale,max_threshold,normalize_volume

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../airlab/airlab/")))) #add airlab dir to working dir

import airlab as al


################ MAIN LOADER FUNCTION ####################

def prepare_vols_multichan(test_path_list,
                       test_mask_path_list,
                       atlas_path_list,
                       atlas_mask_path_list,
                       resample_factor,
                       ismask=True,
                       flip=True,
                       resolution_flip=True,
                       res_atlas=0.5,
                       res_test=0.75,
                       r_test_base=0.75,
                       resample_input=False,
                       speed_crop=False,
                       rotate_atlas=True,
                       tolerance=[50, 50],
                       loss_region_sigma=11.0):

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


    if resample_input is not True:
        res_test_base = res_test
    else:
        res_test_base = r_test_base

    test_list = []
    atlas_list = []
    test_masks_array_list = []
    atlas_masks_array_list = []

    for i in range(0,n_channels):
        test_foo,dummy_affine = load_nifti(test_path_list[i], return_img=False)
        atlas_foo,_ = load_nifti(atlas_path_list[i], return_img=False)

        test_foo = normalize_volume(test_foo)
        atlas_foo = normalize_volume(atlas_foo)

        test_list.append(test_foo)
        atlas_list.append(atlas_foo)

    for i in range(0,len(test_mask_path_list)):
        test_mask,_ = load_nifti(test_mask_path_list[i], return_img=False)
        atlas_mask,_ = load_nifti(atlas_mask_path_list[i], return_img=False)

        atlas_masks_array_list.append(atlas_mask)
        test_masks_array_list.append(test_mask)


    if resample_input is True:
        print("resampling to: ", res_test, " mm")
        for i in range(0,n_channels):
            test_list[i] = resample(test_list[i],res_atlas,res_test)

        #resample and threshold the masks
        for i in range(0,len(test_mask_path_list)):
            test_masks_array_list[i] = resample(test_masks_array_list[i],res_atlas,res_test)
            test_masks_array_list[i] = mean_threshold(test_masks_array_list[i])


    if rotate_atlas is True:
        for i in range(0,n_channels):
            atlas_list[i] = np.rot90(atlas_list[i],k=3,axes=(1, 2))
        for i in range(0,len(test_mask_path_list)):
            atlas_masks_array_list[i] = np.rot90(atlas_masks_array_list[i],k=3,axes=(1, 2))
            atlas_masks_array_list[i] = max_threshold(atlas_masks_array_list[i])
    else:
        atlas_masks_array_list[i] = max_threshold(atlas_masks_array_list[i])




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
    for i in range(0,len(test_mask_path_list)):
        atlas_mask1_reshaped, test_mask1_reshaped, mask_p_matrix = res_match_and_rescale(atlas_masks_array_list[i],
                                                                                    test_masks_array_list[i],
                                                                                    res_atlas,
                                                                                    res_test_base,
                                                                                    resample_factor,
                                                                                    res_flip=resolution_flip)
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


        if speed_crop:
            test_mask1_reshaped_foo,atlas_mask1_reshaped,_,_,crop_matrix1,crop_matrix2 = crop_around_COM(test_mask1_reshaped,
                                                                                                      aligned_atlas_mask_reshaped,
                                                                                                      test_masks_list_reshaped_cropped_aligned[0],
                                                                                                      atlas_masks_list_reshaped_cropped_aligned[0],
                                                                                                      tolerance)

        test_mask1_reshaped = test_mask1_reshaped_foo
        ## binarize and clean masks
        test_mask1_reshaped = max_threshold(test_mask1_reshaped)
        where_objs = morphology.remove_small_objects(test_mask1_reshaped.astype(np.bool_), 30)
        test_mask1_reshaped[where_objs < 0.5] = 0
        test_mask1_reshaped = gaussian_filter(test_mask1_reshaped, sigma=0.1)
        atlas_mask1_reshaped = gaussian_filter(atlas_mask1_reshaped, sigma=0.1)

        test_mask1_reshaped = max_threshold(test_mask1_reshaped)
        atlas_mask1_reshaped = max_threshold(atlas_mask1_reshaped)


        [s0, s1, s2] = test_mask1_reshaped.shape

        ### the loss region will always be the first mask pair in the list!
        if counter == 0:
            fixed_loss_region = gaussian_filter(test_mask1_reshaped, sigma=loss_region_sigma)
            moving_loss_region = gaussian_filter(atlas_mask1_reshaped, sigma=loss_region_sigma)
            fixed_loss_region[fixed_loss_region > 0] = 1
            moving_loss_region[moving_loss_region > 0] = 1

            fixed_loss_region = al.Image(fixed_loss_region, [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)
            moving_loss_region = al.Image(moving_loss_region, [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)



        fixed_mask = al.Image(test_mask1_reshaped, [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)
        moving_mask = al.Image(atlas_mask1_reshaped, [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype)
        fixed_mask_list.append(fixed_mask)
        moving_mask_list.append(moving_mask)

        counter += 1



    fixed_image_list = []
    moving_image_list = []
    for i in range(0,n_channels):
        atlas_reshaped, test_reshaped, test_p_matrix = res_match_and_rescale(atlas_list[i],
                                                                            test_list[i],
                                                                            res_atlas,
                                                                            res_test_base,
                                                                            resample_factor,
                                                                            res_flip=resolution_flip)

        atlas_reshaped_aligned,_ = align_COM_masks(test_reshaped,
                                                atlas_reshaped,
                                                test_masks_list_reshaped[0],
                                                atlas_masks_list_reshaped[0])


        if speed_crop:
            test_reshaped_cropped_foo,atlas_reshaped_cropped_foo,_,_,crop_matrix1,crop_matrix2 = crop_around_COM(test_reshaped,
                                                                                                        atlas_reshaped_aligned,
                                                                                                        test_masks_list_reshaped_cropped_aligned[0],
                                                                                                        atlas_masks_list_reshaped_cropped_aligned[0],
                                                                                                        tolerance)

        test_reshaped_cropped = gaussian_filter(test_reshaped_cropped_foo,sigma=0.1)
        fixed_image_list.append(al.Image(test_reshaped_cropped.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype))
        moving_image_list.append(al.Image(atlas_reshaped_cropped_foo.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0],dtype=ddtype))





    return fixed_image_list,moving_image_list,fixed_mask_list,moving_mask_list,fixed_loss_region,moving_loss_region,test_p_matrix
