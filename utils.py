import os,sys
import argparse


def print_no_newline(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def parse_args_mrtrix():
    parser = argparse.ArgumentParser(description="Prepares data to run probibalistic tractography. Checks ROI inputs for matching geometry.")
    #------------------- Required Arguments -------------------
    parser.add_argument('-c','--caselist', help="Partial case list", type=str, required=True)
    parser.add_argument('-b','--basepath', help="case case directory", type=str, required=True)
    return parser.parse_args()
