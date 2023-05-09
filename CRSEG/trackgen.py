import os, sys, argparse, datetime, traceback
import shutil
import math
import random
import string
import traceback
import sys
import numpy as np
import multiprocessing as mp
from utils import print_no_newline, parse_args_mrtrix, count_shells, get_header_resolution, tractography_mask


"""
MrTrix-based probabalistic tractography preprocessing pipeline for brainstem WM bundles

Usage:


Author:
Mark D. Olchanyi -- 03.17.2023
"""



##Parser
args = parse_args_mrtrix()
#------------------- Required Arguments -------------------
basepath = args.basepath
local_data_path = args.datapath
bval_path = args.bvalpath
bvec_path = args.bvecpath
case_list_txt = args.caselist
crop_size = args.cropsize
output_folder = args.output
dcm_json_header_path = args.json_header_path
fsl_preprocess = args.fsl_preprocess

case_list_full = []

casefile = open(case_list_txt, "r")

lines = casefile.readlines()
for index, line in enumerate(lines):
    case = line.strip()
    case_list_full.append(os.path.join(basepath + case))
casefile.close()

for case_path in case_list_full:

    try:
        print("============================================================")
        print("STARTING: ", case_path)
        print("============================================================")


        letters = string.ascii_lowercase
        scratch_str = "temp_" + ''.join(random.choice(letters) for i in range(10))
        scratch_dir = os.path.join(case_path,scratch_str,"")
        output_dir = os.path.join(case_path,output_folder,"")
        print("All final MRTrix volumes will be dropped in ", output_dir)
        if os.path.exists(os.path.join(output_dir,"tracts_concatenated_1mm_cropped.mif")):
            print("MRTRIX outputs already exit...skipping")
            continue

        print("creating temporary scratch directory ", scratch_dir)
        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir)
        if not os.path.exists(output_dir):
            print("making fresh output directory...")
            os.makedirs(output_dir)
        else:
            print("cleaning out existing output directory...")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)

        if not os.path.exists(os.path.join(scratch_dir,"data.nii.gz")):
            case = os.path.basename(case_path)
            os.system("rsync -av " + os.path.join(args.basepath,os.path.basename(case_path),args.datapath) + " " + os.path.join(scratch_dir,"data.nii.gz"))
        if not os.path.exists(os.path.join(scratch_dir,"dwi.bval")):
            os.system("rsync -av " + os.path.join(args.basepath,os.path.basename(case_path),args.bvalpath) + " " + os.path.join(scratch_dir,"dwi.bval"))
        if not os.path.exists(os.path.join(scratch_dir,"dwi.bvec")):
            os.system("rsync -av " + os.path.join(args.basepath,os.path.basename(case_path),args.bvecpath) + " " + os.path.join(scratch_dir,"dwi.bvec"))



        ##### Initial MRTRIX calls #####

        ## perform wrapped FSL preprocessing, requires a DICOM header json to get phase-encoding direction
        if fsl_preprocess:
            assert dcm_json_header_path != None, 'No DICOM header json file provided. This is required for FSL preprocessing!'
            print("---------- STARTING FSL PREPROCESSING (Eddy + Motion Correction) -----------")
            os.system("dwifslpreproc " + os.path.join(scratch_dir,"data.nii.gz") + " " + os.path.join(scratch_dir,"dwi.mif") + " -json_import " + dcm_json_header_path + " -rpe_header -fslgrad " + os.path.join(scratch_dir,"dwi.bvec") + " " + os.path.join(scratch_dir,"dwi.bval"))
            print("Finished FSL preprocessing!")

        ## convert dwi to MIF format
        if not os.path.exists(os.path.join(scratch_dir,"dwi.mif")) and not fsl_preprocess:
            os.system("mrconvert " + os.path.join(scratch_dir,"data.nii.gz") + " " + os.path.join(scratch_dir,"dwi.mif") + " -fslgrad " + os.path.join(scratch_dir,"dwi.bvec") + " " + os.path.join(scratch_dir,"dwi.bval") + " -force")


        # extract header voxel resolution and match it to HCP data (1.25mm iso) and
        # find out if single-shell or not to degermine which FOD algorithm to use.
        os.system("mrinfo -json_all " + os.path.join(scratch_dir,"header.json") + " " + os.path.join(scratch_dir,"dwi.mif") + " -force")
        vox_resolution = get_header_resolution(os.path.join(scratch_dir,"header.json"))
        print("header resolution is " + str(vox_resolution) + " mm")
        shell_count = count_shells(os.path.join(scratch_dir,"header.json"))
        single_shell = shell_count <= 2
        print("...single_shell mode is " + str(single_shell))

        if (vox_resolution > 1.3) or (vox_resolution < 1.2):
            print_no_newline("Resolution is out of bounds!! Regridding dwi to HCP1200 resolution of 1.25mm iso...")
            os.system("mrgrid " + os.path.join(scratch_dir,"dwi.mif") + " regrid -vox 1.25 " + os.path.join(scratch_dir,"dwi_regridded_HCP.mif") + " -force")
            os.system("rm " + os.path.join(scratch_dir,"dwi.mif"))
            os.system("mv " + os.path.join(scratch_dir,"dwi_regridded_HCP.mif") + " " + os.path.join(scratch_dir,"dwi.mif"))
            print("done")
        ## extract mean b0 volume
        print_no_newline("extracting temporary b0...")
        if not os.path.exists(os.path.join(scratch_dir,"mean_b0.mif")):
            os.system("dwiextract " + os.path.join(scratch_dir,"dwi.mif") + " - -bzero | mrmath - mean " + os.path.join(scratch_dir,"mean_b0.mif") + " -axis 3 -force")
            os.system("mrconvert " + os.path.join(scratch_dir,"mean_b0.mif") + " " + os.path.join(scratch_dir,"mean_b0.nii.gz") + " -force")
        print("done")


        ##### SAMSEG CALLS #####

        samseg_path = os.path.join(case_path,"samseg_labels","")
        if not os.path.exists(samseg_path + "seg.mgz"):
            os.system("run_samseg -i " + os.path.join(scratch_dir,"mean_b0.nii.gz") + " -o " + samseg_path + " --threads 8")


        thal_labels = [10,49]
        DC_labels = [28,60]
        cort_labels = [18,54]
        CB_labels = [7,46]
        brainstem_label = 16

        print_no_newline("extracting subcortical samseg labels...")
        ## special dilation and overlap with dilated cortical label to get most anterior portion of DC
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz ") + " --o " + os.path.join(scratch_dir,"DC.nii") + " --match " + str(DC_labels[0]) + " " + str(DC_labels[1]))
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz ") + " --o " + os.path.join(scratch_dir,"cort.nii") + " --match " + str(cort_labels[0]) + " " + str(cort_labels[1]))
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz ") + " --o " + os.path.join(scratch_dir,"thal.nii") + " --match " + str(thal_labels[0]) + " " + str(thal_labels[1]))
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz ") + " --o " + os.path.join(scratch_dir,"CB.nii") + " --match " + str(CB_labels[0]) + " " + str(CB_labels[1]))
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz ") + " --o " + os.path.join(scratch_dir,"brainstem.nii") + " --match " + str(brainstem_label))
        print("done")


        ## get centroid voxel coordinates of union of thalamic and brainstem masks to obtain bounding box location for smaller ROI for the U-net.
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz") + " --o " + os.path.join(scratch_dir,'thal_brainstem_union.mgz') + " --match " + str(brainstem_label) + " " + str(thal_labels[0]) + " " + str(thal_labels[1]))
        os.system("mri_binarize --noverbose --i " + os.path.join(samseg_path,"seg.mgz ") + " --o " + os.path.join(scratch_dir,"all_labels.nii") + " --match " + str(thal_labels[0]) + " " + str(thal_labels[1]) + " " + str(DC_labels[0]) + " " + str(DC_labels[1]) + " " + str(CB_labels[0]) + " " + str(CB_labels[1]) + " " + str(brainstem_label))
        os.system("mrcentroid -voxelspace " + os.path.join(scratch_dir,'thal_brainstem_union.mgz') + " > " + os.path.join(scratch_dir,"thal_brainstem_cntr_coords.txt"))
        os.system("mri_info --vox2ras " + os.path.join(scratch_dir,'thal_brainstem_union.mgz') + " > " + os.path.join(scratch_dir,"thal_brainstem_vox2ras.txt"))

        brainstem_cntr_arr = np.loadtxt(os.path.join(scratch_dir,"thal_brainstem_cntr_coords.txt"))
        brainstem_cntr_arr_hom = np.append(brainstem_cntr_arr,1.0) ## make homogenous array
        vox2ras_mat = np.loadtxt(os.path.join(scratch_dir,"thal_brainstem_vox2ras.txt"))

        ras_cntr = np.matmul(vox2ras_mat,brainstem_cntr_arr_hom.T) ## RAS coordinate transform through nultiplying by transform matrix

        print_no_newline("creating tractography mask from thal and brainstem labels...")
        track_mask = tractography_mask(os.path.join(scratch_dir,"all_labels.nii"),os.path.join(scratch_dir,'tractography_mask.nii.gz'))
        print("done")
        ## extract brain mask
        print_no_newline("extracting brain mask...")
        if not os.path.exists(os.path.join(scratch_dir,"brain_mask.mif")):
            os.system("dwi2mask " + os.path.join(scratch_dir,"dwi.mif") + " " + os.path.join(scratch_dir,"brain_mask.mif") + " -force")
        print("done")
        ## get region-specific response functions to better estimate ODF behavior
        print_no_newline("obtaining response functions...")
        if not os.path.exists(os.path.join(scratch_dir,"wm.txt")):
            if not single_shell:
                os.system("dwi2response dhollander " + os.path.join(scratch_dir,"dwi.mif") + " " + os.path.join(scratch_dir,"wm.txt") + " " + os.path.join(scratch_dir,"gm.txt") + " " + os.path.join(scratch_dir,"csf.txt") + " -force -quiet")
            else:
                os.system("dwi2response tournier " + os.path.join(scratch_dir,"dwi.mif") + " " + os.path.join(scratch_dir,"wm.txt") + " -force -quiet")
        print("done")
        ## fit ODFs
        print_no_newline("fitting ODFs with msmt_CSD...")
        if not os.path.exists(os.path.join(scratch_dir,"wmfod.mif")):
            if not single_shell:
                os.system("dwi2fod msmt_csd " + os.path.join(scratch_dir,"dwi.mif") + " -mask " + os.path.join(scratch_dir,"brain_mask.mif") + " " + os.path.join(scratch_dir,"wm.txt") + " " + os.path.join(scratch_dir,"wmfod.mif") +  " " + os.path.join(scratch_dir,"gm.txt") + " " + os.path.join(scratch_dir,"gmfod.mif") +  " " + os.path.join(scratch_dir,"csf.txt") + " " + os.path.join(scratch_dir,"csffod.mif") + " -force -nthreads 10 -quiet")
            else:
                os.system("dwi2fod csd " + os.path.join(scratch_dir,"dwi.mif") + " " + os.path.join(scratch_dir,"wm.txt") + " " + os.path.join(scratch_dir,"wmfod.mif") + " -mask " + os.path.join(scratch_dir,"brain_mask.mif") + " -lmax 6 -force -nthreads 10 -quiet")
        print("done")
        ## normalize ODFs for group tests
        if not os.path.exists(os.path.join(scratch_dir,"wmfod_norm.mif")):
            if not single_shell:
                os.system("mtnormalise " + os.path.join(scratch_dir,"wmfod.mif") + " " + os.path.join(scratch_dir,"wmfod_norm.mif") + " " + os.path.join(scratch_dir,"gmfod.mif") + " " + os.path.join(scratch_dir,"gmfod_norm.mif") + " " + os.path.join(scratch_dir,"csffod.mif") + " " + os.path.join(scratch_dir,"csffod_norm.mif") + " -mask " + os.path.join(scratch_dir,"brain_mask.mif") + " -force")
            else:
                os.system("mtnormalise " + os.path.join(scratch_dir,"wmfod.mif") + " " + os.path.join(scratch_dir,"wmfod_norm.mif") + " -mask " + os.path.join(scratch_dir,"brain_mask.mif") + " -force")
        ##### convert ROIs into MIFs #####
        if not os.path.exists(os.path.join(scratch_dir,"brainstem.mif")) or not os.path.exists(os.path.join(scratch_dir,"CB.mif")) or not os.path.exists(os.path.join(scratch_dir,"DC.mif")) or not os.path.exists(os.path.join(scratch_dir,"thal.mif")):
            os.system("mrconvert " + os.path.join(scratch_dir,"thal.nii") + " " + os.path.join(scratch_dir,"thal.mif") + " -force")
            os.system("mrconvert " + os.path.join(scratch_dir,"DC.nii") + " " + os.path.join(scratch_dir,"DC.mif") + " -force")
            os.system("mrconvert " + os.path.join(scratch_dir,"cort.nii") + " " + os.path.join(scratch_dir,"cort.mif") + " -force")
            os.system("mrconvert " + os.path.join(scratch_dir,"CB.nii") + " " + os.path.join(scratch_dir,"CB.mif") + " -force")
            os.system("mrconvert " + os.path.join(scratch_dir,"brainstem.nii") + " " + os.path.join(scratch_dir,"brainstem.mif") + " -force")

            print_no_newline("performing intersection of dilated amyg and DC SAMSEG labels...")
            morpho_amount = int(5/vox_resolution)
            print("morphing by " + str(morpho_amount) + " voxels")

            os.system("maskfilter " + os.path.join(scratch_dir,"DC.mif") + " dilate -npass " + str(morpho_amount) + " " + os.path.join(scratch_dir,"DC.mif") + " -force")
            os.system("maskfilter " + os.path.join(scratch_dir,"cort.mif") + " dilate -npass " + str(morpho_amount) + " " + os.path.join(scratch_dir,"cort.mif") + " -force")
            os.system("maskfilter " + os.path.join(scratch_dir,"thal.mif") + " erode -npass " + str(morpho_amount) + " " + os.path.join(scratch_dir,"thal.mif") + " -force")
            os.system("maskfilter " + os.path.join(scratch_dir,"CB.mif") + " erode -npass " + str(morpho_amount) + " " + os.path.join(scratch_dir,"CB.mif") + " -force")
            os.system("mrcalc " + os.path.join(scratch_dir,"DC.mif") + " " + os.path.join(scratch_dir,"cort.mif") + " -mult " + os.path.join(scratch_dir,"DC.mif") + " -force")
            print("done")

        ##### probabilistic tract generation #####
        if not os.path.exists(os.path.join(scratch_dir,"tracts_thal.tck")):
            print("starting tracking on thal.mif")
            os.system("tckgen -algorithm iFOD2 -angle 50 -select 100000 -seed_image " + os.path.join(scratch_dir,"thal.mif") + " -include " + os.path.join(scratch_dir,"brainstem.mif") + " " + os.path.join(scratch_dir,"wmfod_norm.mif") + " " + os.path.join(scratch_dir,"tracts_thal.tck") + " -mask " + os.path.join(scratch_dir,'tractography_mask.nii.gz') + " -max_attempts_per_seed 750 -trials 750 -force -nthreads 10")
        if not os.path.exists(os.path.join(scratch_dir,"tracts_DC.tck")):
            print("starting tracking on DC.mif")
            os.system("tckgen -algorithm iFOD2 -angle 50 -select 100000 -seed_image " + os.path.join(scratch_dir,"DC.mif") + " -include " + os.path.join(scratch_dir,"brainstem.mif") + " -exclude " + os.path.join(scratch_dir,"CB.mif") + " " + os.path.join(scratch_dir,"wmfod_norm.mif") + " " + os.path.join(scratch_dir,"tracts_DC.tck") + " -mask " + os.path.join(scratch_dir,'tractography_mask.nii.gz') + " -max_attempts_per_seed 750 -trials 750 -force -nthreads 10")
        if not os.path.exists(os.path.join(scratch_dir,"tracts_CB.tck")):
            print("starting tracking on CB.mif")
            os.system("tckgen -algorithm iFOD2 -angle 50 -select 50000 -seed_image " + os.path.join(scratch_dir,"CB.mif") + " -include " + os.path.join(scratch_dir,"DC.mif") + " " + os.path.join(scratch_dir,"wmfod_norm.mif") + " " + os.path.join(scratch_dir,"tracts_CB.tck") + " -mask " + os.path.join(scratch_dir,'tractography_mask.nii.gz') + " -max_attempts_per_seed 750 -trials 750 -force -nthreads 10")

        ##### converting tracts into scalar tract densities
        if not os.path.exists(os.path.join(scratch_dir,"tracts_thal.mif")):
            os.system("tckmap " + os.path.join(scratch_dir,"tracts_thal.tck") + " -template " + os.path.join(scratch_dir,"mean_b0.mif") + " -contrast tdi " + os.path.join(scratch_dir,"tracts_thal.mif") + " -force")
        if not os.path.exists(os.path.join(scratch_dir,"tracts_DC.mif")):
            os.system("tckmap " + os.path.join(scratch_dir,"tracts_DC.tck") + " -template " + os.path.join(scratch_dir,"mean_b0.mif") + " -contrast tdi " + os.path.join(scratch_dir,"tracts_DC.mif") + " -force")
        if not os.path.exists(os.path.join(scratch_dir,"tracts_CB.mif")):
            os.system("tckmap " + os.path.join(scratch_dir,"tracts_CB.tck") + " -template " + os.path.join(scratch_dir,"mean_b0.mif") + " -contrast tdi " + os.path.join(scratch_dir,"tracts_CB.mif") + " -force")

        ### perform tract-wise histrogram normalization. matched with the CB tract map. Syntax is | type | input | target | output
        os.system("mrhistmatch linear " + os.path.join(scratch_dir,"tracts_CB.mif") + " " + os.path.join(scratch_dir,"tracts_thal.mif") + " " + os.path.join(scratch_dir,"tracts_CB_matched.mif") + " -force")
        os.system("mrhistmatch linear " + os.path.join(scratch_dir,"tracts_DC.mif") + " " + os.path.join(scratch_dir,"tracts_thal.mif") + " " + os.path.join(scratch_dir,"tracts_DC_matched.mif") + " -force")

        ### turn into color map
        os.system("mrcat " + os.path.join(scratch_dir,"tracts_thal.mif") + " " + os.path.join(scratch_dir,"tracts_CB_matched.mif") + " " + os.path.join(scratch_dir,"tracts_DC_matched.mif") + " " + os.path.join(scratch_dir,"tracts_concatenated.mif") + " -force")
        os.system("mrcolour " + os.path.join(scratch_dir,"tracts_concatenated.mif") + " rgb " + os.path.join(scratch_dir,"tracts_concatenated_color.mif") + " -force")

        ### move relevent files back to static directory
        os.system("mv " + os.path.join(scratch_dir,"tracts_concatenated.mif") + " " + output_dir)
        #os.system("mv " + os.path.join(scratch_dir,"tracts_concatenated_color.mif") + " " + output_dir)
        #os.system("mrconvert " + os.path.join(output_dir,"tracts_concatenated.mif") + " " + os.path.join(output_dir,"tracts_concatenated.nii.gz") + " -datatype float32")
        #os.system("mrconvert " + os.path.join(output_dir,"tracts_concatenated_color.mif") + " " + os.path.join(output_dir,"tracts_concatenated_color.nii.gz") + " -datatype float32")

        #os.system("mrgrid " + os.path.join(output_dir,"tracts_concatenated.mif") + " regrid -voxel 1.0 " + os.path.join(output_dir,"tracts_concatenated_1mm.mif" + " -force"))
        os.system("mri_convert --crop " + str(round(float(brainstem_cntr_arr[0]))) + " " + str(round(float(brainstem_cntr_arr[1]))) + " " +  str(round(float(brainstem_cntr_arr[2]))) + " --cropsize " + crop_size + " " + crop_size + " " + crop_size + " " + os.path.join(scratch_dir,'thal_brainstem_union.mgz') + " " + os.path.join(scratch_dir,'thal_brainstem_union_cropped.mgz'))
        os.system("mrgrid " + os.path.join(output_dir,"tracts_concatenated.mif") + " regrid -template " + os.path.join(scratch_dir,'thal_brainstem_union_cropped.mgz') + " -voxel 1.0 " + os.path.join(output_dir,"tracts_concatenated_1mm_cropped.mif" + " -force"))

        #### delete scratch directory
        print_no_newline("deleting scratch directory...")
        shutil.rmtree(scratch_dir)
        print("done")
        print("finished case mrtrix and fsl preprocessing \n\n\n\n")

    except:
        traceback.print_exc()
        print("some exception has occured!!!!")
        print_no_newline("deleting scratch directory...")
        shutil.rmtree(scratch_dir)
        print("done")
        continue
