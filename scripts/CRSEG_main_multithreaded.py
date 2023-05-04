import sys, shutil, os, math, argparse
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from crseg_loader import prepare_vols_multichan
from crseg_register import register_multichan, propagate
from crseg_utils import affine_trans
from utils import parse_args_crseg_main
import multiprocessing as mp
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../airlab/airlab/")))) #add airlab dir to working dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../CRSEG/"))))

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
      fa_atlas_path = args.atlas_fa_directory
      b0_atlas_path = args.atlas_lowb_directory
      labels_path = args.atlas_aan_label_directory
      WM_seg_path = args.wm_seg_path
      fp = Path(str(args.target_list))
      lines = fp.read_text().splitlines()
      cases = []
      for i in range(0,len(lines)):
            cases.append(lines[i].split('/')[-1])
      basepath = lines[0].replace(lines[0].split('/')[-1],'')


      ####################################################

      ##### script to check resampling dice scores
      ######## SET RELEVANT PATHS ##########

      ############## LIST OF WM MASKS #################

      similarity_file_names = "similarity_files/"
      output_folder_name = "/CRSEG_AAN_NUCLEI_v16/"
      atlas_path_list = [fa_atlas_path,b0_atlas_path]
      atlas_mask_path_list = []
      WM_LANDMARK_LIST = ["RN","TECTUM","CTG","CST","SCP_ROSTRAL","MLF","ML"]
      for lm in WM_LANDMARK_LIST:
          atlas_mask_path_list.append(atlases_path + "/superstructures_v2/" + lm + ".nii")


      print("RUNNING CRSEG ON CASE LIST: ", cases)
      print("\n ---------------------------------------------------------------")
      print("Full basepath is: ", basepath)
      print("Atlas channel list is: ", atlas_path_list)
      print("Atlas mask list is: ", atlas_mask_path_list)
      print("\n ---------------------------------------------------------------")



      ###############################################################
      ###############################################################


      """
        - pre_affine_step: only set to true, if you do not have an affine transformation
        already present, and you need to run one
        - only_propagate: Does not run the registration phase of CRSEG and only progagates
          along a saved displacement field
      """

      penalty = "NGF_CHAN"
      is_mask = True
      using_landmarks = True
      pre_affine_step= False            # run affine if pre-registered volumes do not exist!
      only_propagate = False
      multiplier = 1.0
      rf = None
      pyramid_ds=[2,2,2]
      n_iters=[50, 50, 50, 120]
      regularisation_weight = [30,40,40,500]
      step_size = [4e-3, 3e-3, 2e-3, 1e-3]
      mesh_spacing = [3,3,3,3]
      weights = [5000,10000,20000,40000]
      weights_raw = np.ones(len(atlas_mask_path_list))
      crop=True
      crop_tolerance=[45,45]


      ##############################################################
      ##############################################################


      case_list = list(split(cases,cpu_num))
      case_list = np.array(case_list)
      print("Length of provided case list is: ", len(case_list))
      print("case list is: ", case_list)
      print("AAN label path is: ", labels_path)
      print("--------------------------------------------------")

      for case in cases:
            print("Starting ", case)
            test_mask_path_list = []

            test_path_list = [basepath + case + "/fa.nii.gz", basepath + case + "/b0.nii.gz"] # changed to null gradient compared to atlas T2

            scratch_dir = os.path.join(basepath,case,"scratch")

            if not os.path.isdir(scratch_dir):
                os.makedirs(scratch_dir)

            if pre_affine_step:
                if os.path.exists(os.path.join(scratch_dir,"affine_mat.txt")) and os.path.exists(scratch_dir,"inverse_affine_mat.txt") and os.path.exists(scratch_dir,"b0_affine_transformed.nii.gz") and os.path.exists(scratch_dir,"b0_affine_transformed.nii.gz"):
                    print("found affine and inverse affine transforms...will use these")
                else:
                    print("starting affine step...")
                    affine_trans(atlas_path_list,test_path_list,test_mask_path,scratch_dir)
            else:
                print("--------------- IGNORING AFFINE STEP...but still searching for one -----------------------")
                os.system("reg_transform -invAff " + basepath + case + "/scratch/affine_mat.txt " + basepath + case + "/scratch/inverse_affine_mat.txt")



            #### REDO THIS
            test_path1 = basepath + case + "/scratch/fa.nii"
            test_path2 = basepath + case + "/scratch/b0.nii"      # changed to null gradient compared to atlas T2
            test_path_list = [test_path1,test_path2]

            test_mask_path_list = []
            for lm in WM_LANDMARK_LIST:
                test_mask_path_list.append(basepath + case + "/scratch/" + lm + ".nii")


            _,dummy_affine = load_nifti(test_path1, return_img=False)
            resample_resolution=res

            fixed_image_list,moving_image_list,fixed_mask_list,moving_mask_list,fixed_loss_region,moving_loss_region,test_p_matrix = prepare_vols_multichan(test_path_list,
                                                                                                                                                        test_mask_path_list,
                                                                                                                                                        atlas_path_list,
                                                                                                                                                        atlas_mask_path_list,
                                                                                                                                                        resample_factor=rf,
                                                                                                                                                        pre_affine_step=pre_affine_step,
                                                                                                                                                        ismask = True,
                                                                                                                                                        flip=False,
                                                                                                                                                        resolution_flip=True,
                                                                                                                                                        res_atlas=0.5,
                                                                                                                                                        res_test=0.5,
                                                                                                                                                        r_test_base=None,
                                                                                                                                                        resample_input=False,
                                                                                                                                                        speed_crop=True,
                                                                                                                                                        tolerance=crop_tolerance)



            save_nifti(basepath + case + "/scratch" + "/f_im_cropped.nii.gz",fixed_image_list[1].numpy(),dummy_affine)
            save_nifti(basepath + case + "/scratch" + "/m_im_cropped.nii.gz",moving_image_list[1].numpy(),dummy_affine)
            for i in range(len(fixed_mask_list)):
                save_nifti(basepath + case + "/scratch" + "/f_mask" + str(i) + ".nii.gz",fixed_mask_list[i].numpy(),dummy_affine)
                save_nifti(basepath + case + "/scratch" + "/m_mask" + str(i) + ".nii.gz",moving_mask_list[i].numpy(),dummy_affine)


            savepath = basepath + case + output_folder_name
            savepath_normalized = savepath[:-1] + "_normalized/"

            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            if not os.path.isdir(savepath_normalized):
                os.makedirs(savepath_normalized)
            else:
                shutil.rmtree(savepath_normalized)
                os.makedirs(savepath_normalized)


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



            save_nifti(basepath + case + '/scratch/warped_original_volume.nii.gz',warped_test_image.numpy(),np.eye(4))
            _,dummy_affine = load_nifti(test_path1, return_img=False)
            foo = 0


            propagate(labels_path,
                  savepath,
                  basepath + case,
                  savepath_normalized,
                  displacement_compound_thalamic,
                  foo,
                  test_mask_path_list[0],
                  atlas_mask_path_list[0],
                  dummy_affine,
                  resample_factor=rf,
                  r_atlas=0.5,
                  r_test=0.5,
                  r_test_base=res,
                  resample_input=False,
                  rotate_atlas=False,
                  flip=False,
                  overlap=label_overlap,
                  resolution_flip=True,
                  speed_crop=True,
                  tolerance=crop_tolerance)

            print("\n " + case + " FINISHED!")


if __name__ == '__main__':
    main()
    print("--------------------------------------------")
    print("\n       CRSEG FINISHED! \n")
    print("--------------------------------------------")
