import sys, shutil, os, math, argparse
import numpy as np
from dipy.io.image import load_nifti,save_nifti
import multiprocessing as mp
import nibabel as nib
import nibabel.processing
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../airlab/airlab/")))) #add airlab dir to working dir
sys.path.append('../CRSEG/')

from crseg_loader import prepare_vols_multichan
from crseg_register import register_multichan, propagate
from crseg_utils import affine_trans
from utils import parse_args_crseg_main, print_no_newline
import airlab as al



"""

MAIN CRSEG AAN PROCESSING SCRIPT

requirements:

MD DTI volume
FA DTI volume

Corresponding MD and FA FSL DTI atlas

AirLab placed in the same directory as this script. Otherwise modify the relative path

"""






def split(list_a, cpu_num):
    size = math.ceil(len(list_a)/cpu_num)

    print("--------------------------------------------------------")
    print("             SPAWNING ", str(cpu_num), "PROCESSES")
    print("--------------------------------------------------------")
    print("                TARGET SIZE: ", str(size))
    print("--------------------------------------------------------")
    for i in range(0, len(list_a), size):
        yield list_a[i:i + size]




def main():

      args = parse_args_crseg_main()

      # SET ALL GLOBAL VARS
      global label_overlap
      global atlases_path
      global atlas_path_list
      global atlas_mask_path_list
      global weights_raw
      global weights

      # RETRIEVE ARGS
      cpu_num = int(args.num_threads)
      label_overlap = float(args.label_overlap)
      case_list_raw = args.target_list
      res = float(args.resolution)
      fa_path = args.target_fa_path
      b0_path = args.target_lowb_path
      fa_atlas_path = args.atlas_fa_path
      b0_atlas_path = args.atlas_lowb_path
      labels_path = args.atlas_aan_label_directory
      label_list_path = args.label_list_path
      wm_seg_path = args.wm_seg_path
      atlas_wm_seg_path = args.atlas_wm_seg_path
      output_path = args.output_directory



      print("\n---------------------------------------------------------------")
      print("                        *RUNNING CRSEG*                           ")
      print("---------------------------------------------------------------")


      ### just to extract a couple values
      label_array = np.load(label_list_path) #load the label list from the CNN and ignore the first entry (0=background)
      label_array = np.delete(label_array, 0)
      label_array = np.delete(label_array, -1)

      print("Found label array has the given vals: ", label_array)




      penalty = "NGF_CHAN"
      using_landmarks = True
      pre_affine_step= True            # run affine if pre-registered volumes do not exist!
      multiplier = 1.0
      rf = None
      pyramid_ds=[8,4,2]
      n_iters=[100, 100, 50, 50]
      #n_iters=[2, 2, 2, 2]
      regularisation_weight = [1,5,10,20]
      step_size = [4e-3, 2e-3, 1e-3, 5e-4]
      mesh_spacing = [3,3,3,3]
      weights = [5000,10000,40000,1000000]
      weights_raw = np.ones(label_array.size)
      crop=True
      crop_tolerance=[50,50]


      ##############################################################
      ##############################################################
      print("found " + str(label_array.size) + " WM ROIs as fiducials in: " + label_list_path)

      test_path_list = [fa_path,b0_path] # changed to null gradient compared to atlas T2
      atlas_path_list = [fa_atlas_path,b0_atlas_path]

      scratch_dir = os.path.join(output_path,"scratch")
      if not os.path.exists(output_path):
            print("creating output directory")
            os.makedirs(output_path)
      if not os.path.isdir(scratch_dir):
          print("creating scratch directory")
          os.makedirs(scratch_dir)
      else:
          print("found existing scratch directory")

      """
        - pre_affine_step: only set to true, if you do not have an affine transformation
        already present, and you need to run one
        - only_propagate: Does not run the registration phase of CRSEG and only progagates
          along a saved displacement field
      """

      # conforming step
      fa_test_vol = nib.load(fa_path)
      b0_test_vol = nib.load(b0_path)
      fa_atlas_vol = nib.load(fa_atlas_path)
      b0_atlas_vol = nib.load(b0_atlas_path)
      test_wm_seg_vol = nib.load(wm_seg_path)

      test_header = b0_test_vol.header
      test_wm_seg_header = test_wm_seg_vol.header
      atlas_header = b0_atlas_vol.header
      common_affine_test = b0_test_vol.affine   ### use this for saving everything!!
      common_affine_atlas = b0_atlas_vol.affine   ### use this for saving everything!!

      print_no_newline("conforming test volumes...")
      fa_test_vol = nib.processing.conform(fa_test_vol,out_shape=atlas_header.get_data_shape(), voxel_size=atlas_header.get_zooms(), order=3)
      b0_test_vol = nib.processing.conform(b0_test_vol,out_shape=atlas_header.get_data_shape(), voxel_size=atlas_header.get_zooms(), order=3)
      test_wm_seg_vol = nib.processing.conform(test_wm_seg_vol,out_shape=atlas_header.get_data_shape(), voxel_size=atlas_header.get_zooms(), order=0)
      print(" done")

      nib.save(fa_test_vol,os.path.join(scratch_dir,"fa_test.nii.gz"))
      nib.save(b0_test_vol,os.path.join(scratch_dir,"b0_test.nii.gz"))
      nib.save(b0_atlas_vol,os.path.join(scratch_dir,"b0_atlas.nii.gz"))
      nib.save(test_wm_seg_vol,os.path.join(scratch_dir,"test_wm_seg.nii.gz"))


      if pre_affine_step:
          if os.path.exists(os.path.join(scratch_dir,"affine_mat.txt")) and os.path.exists(os.path.join(scratch_dir,"inverse_affine_mat.txt")) and os.path.exists(os.path.join(scratch_dir,"b0_affine_transformed.nii.gz")) and os.path.exists(os.path.join(scratch_dir,"b0_affine_transformed.nii.gz")):
              print("found affine and inverse affine transforms...will use these")
          else:
              print("starting affine step...")
              affine_trans(atlas_path_list,[os.path.join(scratch_dir,"fa_test.nii.gz"),os.path.join(scratch_dir,"b0_test.nii.gz")],os.path.join(scratch_dir,"test_wm_seg.nii.gz"),scratch_dir)
      else:
          print("--------------- IGNORING AFFINE STEP...but still searching for one -----------------------")
          os.system("reg_transform -invAff " + basepath + case + "/scratch/affine_mat.txt " + basepath + case + "/scratch/inverse_affine_mat.txt")
      test_path_list = [os.path.join(scratch_dir,"fa_affine_transformed.nii.gz"),os.path.join(scratch_dir,"b0_affine_transformed.nii.gz")] #redefine with affine transformed volumes
      wm_seg_path = os.path.join(scratch_dir,"WM_masks_affine_transformed.nii.gz")


      _,dummy_affine = load_nifti(test_path_list[0],return_img=False) #keep this affine transform for everything

      fixed_image_list,moving_image_list,fixed_mask_list,moving_mask_list,fixed_loss_region,moving_loss_region,test_p_matrix = prepare_vols_multichan(test_path_list,
                                                                                                                                                  wm_seg_path,
                                                                                                                                                  atlas_path_list,
                                                                                                                                                  atlas_wm_seg_path,
                                                                                                                                                  scratch_dir,
                                                                                                                                                  label_array,
                                                                                                                                                  save_affine=common_affine_test,
                                                                                                                                                  res_atlas=0.5,
                                                                                                                                                  res_test=1.0,
                                                                                                                                                  speed_crop=True,
                                                                                                                                                  rotate_atlas=False,
                                                                                                                                                  tolerance=crop_tolerance)


      nib.save(nib.Nifti1Image(fixed_image_list[1].numpy(),affine=common_affine_test), os.path.join(scratch_dir,"f_im_cropped.nii.gz"))
      nib.save(nib.Nifti1Image(moving_image_list[1].numpy(),affine=common_affine_test), os.path.join(scratch_dir,"m_im_cropped.nii.gz"))
      for i in range(len(fixed_mask_list)):
          nib.save(nib.Nifti1Image(fixed_mask_list[i].numpy(),affine=common_affine_test), os.path.join(scratch_dir,"f_mask" + str(i) + ".nii.gz"))
          nib.save(nib.Nifti1Image(moving_mask_list[i].numpy(),affine=common_affine_test), os.path.join(scratch_dir,"m_mask" + str(i) + ".nii.gz"))


      if not os.path.isdir(output_path):
          os.makedirs(output_path)
      #weights = [0,0,0,0]
      weight_list = [weights_raw*weights[0],weights_raw*weights[1],weights_raw*weights[2],weights_raw*weights[3]]
      num_iters = n_iters


      print("beginning MSE registration")
      displacement_compound_thalamic, warped_test_image = register_multichan(scratch_dir,
                                                                       fixed_image_list,
                                                                       moving_image_list,
                                                                       fixed_mask_list,
                                                                       moving_mask_list,
                                                                       fixed_loss_region,
                                                                       moving_loss_region,
                                                                       regularisation_weight,
                                                                       n_iters=n_iters,
                                                                       step_size=step_size,
                                                                       pyramid_ds=pyramid_ds,
                                                                       pyramid_sigma=[4,4,4,6],
                                                                       mask_weights=weight_list,
                                                                       mesh_spacing=mesh_spacing,
                                                                       relax_alpha=2.0,
                                                                       relax_beta=1.1,
                                                                       affine_step=False,
                                                                       diffeomorphic=False,
                                                                       using_masks=True,
                                                                       use_single_channel=False,
                                                                       relax_regularization=True,
                                                                       no_superstructs=False,
                                                                       use_varifold=True,
                                                                       varifold_sigma=7.0,
                                                                       use_MSE=False,
                                                                       use_Dice=False)

      nib.save(nib.Nifti1Image(warped_test_image.numpy(),affine=b0_atlas_vol.affine), os.path.join(scratch_dir,'warped_original_volume.nii.gz'))


      propagate(labels_path,
            wm_seg_path,
            atlas_wm_seg_path,
            output_path,
            scratch_dir,
            atlas_header,
            displacement_compound_thalamic,
            save_affines=[common_affine_test,common_affine_atlas],
            resample_factor=rf,
            r_atlas=0.5,
            r_test=0.5,
            r_test_base=res,
            rotate_atlas=False,
            flip=False,
            overlap=label_overlap,
            resolution_flip=True,
            speed_crop=True,
            tolerance=crop_tolerance)


if __name__ == '__main__':
    main()
    print("--------------------------------------------")
    print("\n       CRSEG FINISHED! \n")
    print("--------------------------------------------")
