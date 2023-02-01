import os,sys
import argparse


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
