import os,sys
import argparse
import json
import numpy as np
from dipy.io.image import load_nifti, save_nifti

def print_no_newline(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def parse_args_mrtrix():
    parser = argparse.ArgumentParser(description="Prepares data to run probibalistic tractography. Checks ROI inputs for matching geometry.")
    #------------------- Required Arguments -------------------
    parser.add_argument('-cl','--caselist', help="Partial case list", type=str, default=None, required=False)
    parser.add_argument('-c','--case', help="Single Case, default is none", type=str, default=None, required=False)
    parser.add_argument('-b','--basepath', help="Case case directory", type=str, required=True)
    parser.add_argument('-d','--datapath', help="Local path to original DWI file", type=str, required=True)
    parser.add_argument('-bc','--bvalpath', help="Local path to bval file", type=str, required=True)
    parser.add_argument('-bv','--bvecpath', help="Local path to bvec file", type=str, required=True)
    parser.add_argument('-cs','--cropsize', help="crop size around the brainstem", type=str, required=True)
    parser.add_argument('-o','--output',help="output directory for MRTrix tractography volumes", type=str, required=True)
    parser.add_argument('-hdr','--json_header_path',help="DICOM header json file (generated by something like dcm2niix) from the generated nifti", type=str, default=None, required=False)
    parser.add_argument('-p','--fsl_preprocess',help="preprocess the dwi nitfi, this will perform MrTrix-wrapped FSL eddy and motion correction", type=str, default=False, required=False)
    parser.add_argument('-sc','--scrape',help="Scraping for default DWI, bval and bvec files...not recommended", type=str, default=False, required=False)
    parser.add_argument('-us','--unet_segment',help="Segment out white matter from the tract files", type=str, default=False, required=False)

    return parser.parse_args()


def parse_args_crseg_main():

    parser = argparse.ArgumentParser(description="Runs white-matter constrained segmentation of Ascending arousal network nuclei in any HARDI volume")
    #------------------- Required Arguments -------------------
    parser.add_argument('-targfa','--target_fa_path', help="Directory for target FA volume", type=str, required=True)
    parser.add_argument('-targlowb','--target_lowb_path', help="Directory for target LowB volume", type=str, required=True)
    parser.add_argument('-atfa','--atlas_fa_path', help="Directory for FA atlas", type=str, required=True)
    parser.add_argument('-atlowb','--atlas_lowb_path', help="Directory for LowB atlas", type=str, required=True)
    parser.add_argument('-ataan','--atlas_aan_label_directory', help="Directory where GT atlas AAN labels are stored", type=str, required=True)
    parser.add_argument('-llp','--label_list_path', help="Path to numpy array containing label values", type=str, required=True)
    parser.add_argument('-wm','--wm_seg_path', help="Path to volume with white matter segmentations, must be in LUT format", type=str, required=True)
    parser.add_argument('-atwm','--atlas_wm_seg_path', help="Path to atlas volume with white matter segmentations, must be in LUT format", type=str, required=True)
    parser.add_argument('-o','--output_directory', help="Directory for output of all CRSEG files", type=str, required=True)
    parser.add_argument('-t','--target_list', help="Directory or list of directories containing all target DTI volumes and their derivatives", type=str, required=False)
    parser.add_argument('-r','--resolution', help="Resolution of target volumes", type=str, required=False)
    parser.add_argument('-lo','--label_overlap', help="Overlap percentage of propagated labels. Lower percentage is better for lower-res volumes where more aliasing occurs", type=str, required=True)
    parser.add_argument('-n','--num_threads', help="Number of CPU threads to spawn", type=str, required=True)

    return parser.parse_args()


def count_shells(dwi_json_path):
    f = open(dwi_json_path)
    dw_head = json.load(f)

    shell_list = []
    for enc in dw_head["keyval"]["dw_scheme"]:
        shell_mag_val_true = enc[3]
        ## shells in some data are +- 50 of ideal shell value..this is just a dirty way to
        ## round them off
        shell_mag_val_rounded = int(round(shell_mag_val_true/500)*500)
        shell_list.append(shell_mag_val_rounded)
    n_shells = len((set(shell_list)))
    print("Diffusion data contains " + str(n_shells) + " unique shell values...shell vals are: " + str(list(set(shell_list))))
    return n_shells


def get_header_resolution(dwi_json_path):
    f = open(dwi_json_path)
    dw_head = json.load(f)
    return dw_head['spacing'][0]


def tractography_mask(template_vol_path,output_path):
    template_vol,aff = load_nifti(template_vol_path, return_img=False)
    mask_vol = np.zeros_like(template_vol,dtype=int)

    label_coords=np.argwhere(template_vol==1)

    min_point=np.min(label_coords[:],axis=0)
    max_point=np.max(label_coords[:],axis=0)

    mask_vol[min_point[0]:max_point[0],min_point[1]:max_point[1],min_point[2]:max_point[2]] = 1
    save_nifti(output_path,mask_vol,aff)


def rescale_intensities(vol,factor=5):
    vol = vol - vol.mean()
    vol = vol/(factor*vol.std())
    vol += 0.5
    #clip
    vol[vol < 0] = 0
    vol[vol > 1] = 1

    return vol
