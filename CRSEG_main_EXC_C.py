import sys
import shutil
import os
import math
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from crseg_loader import prepare_vols_multichan
from crseg_register import register_multichan, propagate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("./airlab/airlab/")))) #add airlab dir to working dir

import airlab as al



"""

MAIN CRSEG AAN PROCESSING SCRIPT

requirements:

MD DTI volume
FA DTI volume

Corresponding MD and FA FSL DTI atlas

AirLab placed in the same directory as this script. Otherwise modify the relative path in line 10

"""




##### script to check resampling dice scores


######## SET RELEVANT PATHS ##########
basepath = "../data/C/"
atlases_path = "../Atlases/CRSEG_atlas"
######################################
############## Main input ###############

#### fixed case list

cases = ["C1","C2"]

####




atlas_path1 = atlases_path + "/FA_0.5mm.mgz"
atlas_path2 = atlases_path + "/MD_V2_0.5mm.mgz"

atlas_mask1_path = atlases_path + "/superstructures/CTG.mgz"
atlas_mask2_path = atlases_path + "/superstructures/TECTUM.mgz"
atlas_mask3_path = atlases_path + "/superstructures/SCP_ROSTRAL.mgz"
atlas_mask4_path = atlases_path + "/superstructures/CST.mgz"
atlas_mask5_path = atlases_path + "/superstructures/RN.mgz"


atlas_path_list = [atlas_path1,atlas_path2]
atlas_mask_path_list = [atlas_mask1_path,atlas_mask2_path,atlas_mask3_path,atlas_mask4_path,atlas_mask5_path]



similarity_file_names = "similarity_files/"
penalty = "NGF_CHAN"
is_mask = True
using_landmarks = True

multiplier = 1.0
rf = None

n_iters=[300, 200, 150, 140]
#n_iters=[2, 2, 2, 2]

#regularisation_weight = [0.7,1.2,1.6,2.0]
regularisation_weight = [1e-2, 1e-1, 1e-0, 5]
step_size = [5e-3, 1e-3, 6e-4, 4e-4]
mesh_spacing = [3,4,4,4]

# isotropic weighting array for superstructures
weights_raw = np.ones(len(atlas_mask_path_list))
#weights = [15,20,25,30]
#weights = [3,2,1,0.5]
weights = [0.5,0.3,0.1,0.05]

crop=True
crop_tolerance=[50, 50]
resample_resolution=0.75
res = 0.75



for case in cases:

    print("beginning case: ",case," at resolution: ",res, " mm")

    test_path1 = basepath + case + "/fa.nii.gz"
    test_path2 = basepath + case + "/md.nii.gz"

    test_mask1_path = basepath + case + "/superstructures/CTG.nii"
    test_mask2_path = basepath + case + "/superstructures/TECTUM.nii"
    test_mask3_path = basepath + case + "/superstructures/SCP_ROSTRAL.nii"
    test_mask4_path = basepath + case + "/superstructures/CST.nii"
    test_mask5_path = basepath + case + "/superstructures/RN.nii"

    test_path_list = [test_path1,test_path2]
    test_mask_path_list = [test_mask1_path,test_mask2_path,test_mask3_path,test_mask4_path,test_mask5_path]

    _,dummy_affine = load_nifti(test_path1, return_img=False)
    resample_resolution=res

    fixed_image_list,moving_image_list,fixed_mask_list,moving_mask_list,fixed_loss_region,moving_loss_region,test_p_matrix = prepare_vols_multichan(test_path_list,
                                                                                                                                                test_mask_path_list,
                                                                                                                                                atlas_path_list,
                                                                                                                                                atlas_mask_path_list,
                                                                                                                                                resample_factor=rf,
                                                                                                                                                ismask = True,
                                                                                                                                                flip=False,
                                                                                                                                                resolution_flip=True,
                                                                                                                                                res_atlas=0.5,
                                                                                                                                                res_test=resample_resolution,
                                                                                                                                                r_test_base=res,
                                                                                                                                                resample_input=False,
                                                                                                                                                speed_crop=True,
                                                                                                                                                tolerance=crop_tolerance)

    print("length is: ",len(fixed_image_list))
    print("length is: ",len(moving_image_list))




    if not os.path.isdir(basepath + case + "/scratch"):
        os.makedirs(basepath + case + "/scratch")
    else:
        shutil.rmtree(basepath + case + "/scratch")
        os.makedirs(basepath + case + "/scratch")

    save_nifti(basepath + case + "/scratch" + "/f1_im_cropped.nii.gz",fixed_image_list[0].numpy(),dummy_affine)
    save_nifti(basepath + case + "/scratch" + "/m1_im_cropped.nii.gz",moving_image_list[0].numpy(),dummy_affine)







    #weights = [0,0,0,0]
    weight_list = [weights_raw*weights[0],weights_raw*weights[1],weights_raw*weights[2],weights_raw*weights[3]]

    num_iters = n_iters

    print("beginning MSE registration")
    displacement_compound_thalamic, warped_test_image = register_multichan(basepath + case + '/',
                                                                     fixed_image_list,
                                                                     moving_image_list,
                                                                     fixed_mask_list,
                                                                     moving_mask_list,
                                                                     fixed_loss_region,
                                                                     moving_loss_region,
                                                                     regularisation_weight,
                                                                     n_iters=n_iters,
                                                                     step_size=step_size,
                                                                     pyramid_ds=[8,4,2],
                                                                     pyramid_sigma=[3,3,4,5],
                                                                     mask_weights=weight_list,
                                                                     mesh_spacing=mesh_spacing,
                                                                     relax_alpha=2.0,
                                                                     relax_beta=1.1,
                                                                     affine_step=False,
                                                                     diffeo=True,
                                                                     using_masks=True,
                                                                     use_single_channel=False,
                                                                     relax_regularization=True,
                                                                     no_superstructs=False,
                                                                     use_varifold=False,
                                                                     varifold_sigma=1.0*(1+((resample_resolution-1)*(0.25))),
                                                                     use_MSE=True,
                                                                     use_Dice=False)


    path = "../Atlases/CRSEG_ATLAS/AAN_probabalistic_labels_new/"

    savepath = basepath + case + "/CRSEG_AAN_NUCLEI/"

    _,dummy_affine = load_nifti(test_path1, return_img=False)


    #save_nifti(basepath + case + "/scratch/warped_atlas_v3.nii.gz",warped_test_image.numpy(),dummy_affine)
    #save_nifti(basepath + case + "/scratch/fixed_volume.nii.gz",fixed_image_list[2].numpy(),dummy_affine)

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    propagate(path,
          savepath,
          displacement_compound_thalamic,
          test_p_matrix,
          test_mask1_path,
          atlas_mask1_path,
          dummy_affine,
          resample_factor=rf,
          r_atlas=0.5,
          r_test=resample_resolution,
          r_test_base=res,
          resample_input=False,
          rotate_atlas=True,
          flip=False,
          overlap=0.25,
          resolution_flip=True,
          speed_crop=True,
          tolerance=crop_tolerance)

    print("\n CRSEG FINISHED!")
