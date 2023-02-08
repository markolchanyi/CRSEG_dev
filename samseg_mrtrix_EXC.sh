#source freeurfer for samseg
export FREESURFER_HOME="/usr/local/freesurfer/7.2.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#run mrtrix preprocessing
python trackgen.py --basepath /autofs/space/nicc_003/users/olchanyi/data/HCP100/ --caselist case_list_EXC.txt --datapath T1w/Diffusion/data.nii.gz --bvalpath T1w/Diffusion/bvals --bvecpath T1w/Diffusion/bvecs
