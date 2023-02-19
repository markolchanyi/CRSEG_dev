import os,sys
import argparse
import json


def print_no_newline(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def parse_args_mrtrix():
    parser = argparse.ArgumentParser(description="Prepares data to run probibalistic tractography. Checks ROI inputs for matching geometry.")
    #------------------- Required Arguments -------------------
    parser.add_argument('-c','--caselist', help="Partial case list", type=str, required=True)
    parser.add_argument('-b','--basepath', help="Case case directory", type=str, required=True)
    parser.add_argument('-d','--datapath', help="Local path to original DWI file", type=str, required=True)
    parser.add_argument('-bc','--bvalpath', help="Local path to bval file", type=str, required=True)
    parser.add_argument('-bv','--bvecpath', help="Local path to bvec file", type=str, required=True)
    return parser.parse_args()


def count_shells(dwi_json_path):
    f = open(dwi_json_path)
    dw_head = json.load(f)

    shell_list = []
<<<<<<< HEAD
    shell_count = 0
    with open(bval_path,"r") as f:
        shell_list = set(f.readlines())
        count = len(shell_list)
    return count
=======
    for enc in dw_head["keyval"]["dw_scheme"]:
        shell_mag_val_true = enc[3]
        ## shells in some data are +- 50 of ideal shell value..this is just a dirty way to
        ## round them off
        shell_mag_val_rounded = int(round(shell_mag_val_true/500)*500)
        shell_list.append(shell_mag_val_rounded)
    n_shells = len((set(shell_list)))
    print("Diffusion data contains " + str(n_shells) + " unique shell values...shell vals are: " + str(list(set(shell_list))))
    return n_shells
>>>>>>> a718f5de44fa2f7aa13c83d35aa226f080ea565e


def get_header_resolution(dwi_json_path):
    f = open(dwi_json_path)
    dw_head = json.load(f)
    return dw_head['spacing'][0]
