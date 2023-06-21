#source freeurfer for samseg
export FREESURFER_HOME="/usr/local/freesurfer/7.2.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#run mrtrix preprocessing
python ../CRSEG/trackgen.py --basepath /autofs/space/nicc_003/users/olchanyi/data/HCP_MGH_ADULT --caselist ../text_files/case_list_HCP_MGH.txt --datapath all_shells/data_corr.nii.gz --bvalpath all_shells/bvals --bvecpath all_shells/bvecs --cropsize 64 --output trackgen_outputs
