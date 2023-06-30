#source freeurfer for samseg 
#7.3.0 is the most stable version that does not throw a BLAS error
#and can also run synthSR and brainstem_subfield seg
export FREESURFER_HOME="/usr/local/freesurfer/7.3.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#run mrtrix preprocessing
python ../CRSEG/trackgen.py --datapath /autofs/space/nicc_003/users/olchanyi/data/HCP100/100307/T1w/Diffusion/data.nii.gz --bvalpath /autofs/space/nicc_003/users/olchanyi/data/HCP100/100307/T1w/Diffusion/bvals --bvecpath /autofs/space/nicc_003/users/olchanyi/data/HCP100/100307/T1w/Diffusion/bvecs --cropsize 64 --output /autofs/space/nicc_003/users/olchanyi/data/HCP100/100307/trackgen_outputs
