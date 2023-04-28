import sys, shutil, os, math, argparse
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from crseg_loader import prepare_vols_multichan
from crseg_register import register_multichan, propagate
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

AirLab placed in the same directory as this script. Otherwise modify the relative path in line 10

"""



def parse_args():

    parser = argparse.ArgumentParser(description="Runs white-matter constrained segmentation of Ascending arousal network nuclei in any HARDI volume")
    #------------------- Required Arguments -------------------
    parser.add_argument('-at','--atlas_directory', help="Base with all DTI atlas volumes and their label derivatives", type=str, required=True)
    parser.add_argument('-t','--target_list', help="Directory or list of directories containing all target DTI volumes and their derivatives", type=str, required=True)
    parser.add_argument('-r','--resolution', help="Resolution of target volumes", type=str, required=True)
    parser.add_argument('-o','--label_overlap', help="Overlap percentage of propagated labels. Lower percentage is better for lower-res volumes where more aliasing occurs", type=str, required=True)
    parser.add_argument('-n','--num_threads', help="Number of CPU threads to spawn", type=str, required=True)

    return parser.parse_args()




def CRSEG_bulk_script(basepath,
                        cases,
                        WM_LANDMARK_LIST,
                        atlas_path_list,
                        atlases_path,
                        atlas_mask_path_list,
                        output_folder_name,
                        labels_path,
                        pyramid_ds,
                        weights,
                        weights_raw,
                        similarity_file_names,
                        penalty,is_mask,
                        using_landmarks,
                        multiplier,
                        rf,
                        n_iters,
                        regularisation_weight,
                        step_size,
                        mesh_spacing,
                        crop,
                        crop_tolerance,
                        resample_resolution,
                        res,
                        label_overlap,
                        pre_affine_step,
                        affine_exists=True):

    for case in cases:
          print("Starting ",case," at resolution: ",res, " mm")
          test_mask_path_list = []
          for lm in WM_LANDMARK_LIST:
              test_mask_path_list.append(basepath + case + "/superstructures/" + lm + ".nii")


          test_path1 = basepath + case + "/fa.nii.gz"
          test_path2 = basepath + case + "/b0.nii.gz"      # changed to null gradient compared to atlas T2
          test_path_list = [test_path1,test_path2]

          print("\n -------------------------------------------")
          print("Test channel list is: ", test_path_list)
          print("Test mask list is: ", test_mask_path_list)

          print("\n -------------------------------------------")


          if not os.path.isdir(basepath + case + "/scratch"):
              os.makedirs(basepath + case + "/scratch")
          else:
              if not affine_exists:
                  shutil.rmtree(basepath + case + "/scratch")
                  os.makedirs(basepath + case + "/scratch")
              else:
                  print("affine volumes exist, ignoring directory restructuring")


          if pre_affine_step or not affine_exists:
              print("--------------- RUNNING AFFINE STEP -----------------------")
              os.system("reg_aladin -ref " + atlas_path_list[1] + " -flo " + test_path_list[1] + " -res " + basepath + case + "/scratch/b0.nii" " -aff " + basepath + case + "/scratch/affine_mat.txt")
              os.system("reg_resample -ref " + atlas_path_list[1] + " -flo " + test_path1 + " -res " + basepath + case + "/scratch/fa.nii" + " -trans " + basepath + case + "/scratch/affine_mat.txt" + " -inter 2")
              os.system("reg_resample -ref " + atlas_path_list[1] + " -flo " + test_path2 + " -res " + basepath + case + "/scratch/b0.nii" + " -trans " + basepath + case + "/scratch/affine_mat.txt" + " -inter 2")

              for ind, lm in enumerate(WM_LANDMARK_LIST):
                  os.system("reg_resample -ref " + atlas_path_list[1] + " -flo " + test_mask_path_list[ind] + " -res " + basepath + case + "/scratch/" + lm + ".nii" + " -trans " + basepath + case + "/scratch/affine_mat.txt" + " -inter 0")
              os.system("reg_transform -invAff " + basepath + case + "/scratch/affine_mat.txt " + basepath + case + "/scratch/inverse_affine_mat.txt")
          else:
              print("--------------- AFFINE ALREADY EXISTS...SKIPPING -----------------------")
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

      args = parse_args()

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
      atlases_path = args.atlas_directory
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
      labels_path = atlases_path + "/AAN_probabalistic_labels_new/AAN_probabalistic_labels_custom_v3/"
      atlas_path_list = [atlases_path + "/FA_0.5mm.nii.gz",atlases_path + "/T2_SHIFTED_0.5mm.nii.gz"]
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
      pre_affine_step= False            #run affine if pre-registered volumes do not exist!
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
      processes = []

      if not only_propagate:
          for i in range(0,len(case_list)):
                p = mp.Process(target=CRSEG_bulk_script,
                                        args=(basepath,
                                        case_list[i],
                                        WM_LANDMARK_LIST,
                                        atlas_path_list,
                                        atlases_path,
                                        atlas_mask_path_list,
                                        output_folder_name,
                                        labels_path,
                                        pyramid_ds,
                                        weights,
                                        weights_raw,
                                        similarity_file_names,
                                        penalty,
                                        is_mask,
                                        using_landmarks,
                                        multiplier,
                                        rf,
                                        n_iters,
                                        regularisation_weight,
                                        step_size,
                                        mesh_spacing,
                                        crop,
                                        crop_tolerance,
                                        res,
                                        res,
                                        label_overlap,
                                        pre_affine_step))

          processes.append(p)
          p.start()

          for process in processes:
                process.join()






      else:
          print("=============== ONLY PROPAGATING ==================")
          print("retrieving displacement field")

          displacement = np.load(os.path.join(basepath + case_list[0], "scratch/displacement.npy"))

          print("Atlas AAN labels exist?   ", os.path.isdir(path))
          savepath = basepath + case + output_folder_name
          savepath_normalized = savepath[:-1] + "_normalized/"

          if not os.path.isdir(savepath):
              os.makedirs(savepath)
          if not os.path.isdir(savepath_normalized):
              os.makedirs(savepath_normalized)


          test_path1 = basepath + case_list[0] + "/fa.nii.gz"
          _,dummy_affine = load_nifti(test_path1, return_img=False)
          propagate(labels_path,
                savepath,
                basepath + case,
                savepath_normalized,
                displacement,
                test_p_matrix,
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
