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


def count_shells(bval_path):
    shell_list = []
    shell_count = 0
    with open(bval_path,"r") as f:
        shell_list = set(f.readlines())
        count = len(shell_list)
    return count


def get_header_resolution(dwi_json_path):
    f = open(dwi_json_path)
    dw_head = json.load(f)
    return dw_head['spacing'][0]
