#source freeurfer for samseg
export FREESURFER_HOME="/usr/local/freesurfer/7.2.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#run mrtrix preprocessing
python ../CRSEG/trackgen.py --basepath /autofs/space/nicc_003/users/olchanyi/data/RESPONSE_PILOT_FULL/ --caselist ../text_files/case_list_diffsr.txt --datapath ses-1/dwi/dsistudio/data.nii.gz --bvalpath ses-1/dwi/dsistudio/bvals --bvecpath ses-1/dwi/dsistudio/bvecs --cropsize 64 --output mrtrix_outputs_raw
