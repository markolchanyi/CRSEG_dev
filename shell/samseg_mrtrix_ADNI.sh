#source freeurfer for samseg
export FREESURFER_HOME="/usr/local/freesurfer/7.2.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#run mrtrix preprocessing
python ../CRSEG/trackgen.py --basepath /autofs/space/nicc_003/users/olchanyi/data/ADNI/ADVANCED_037_S_6992/Axial_MB_DTI/2021-11-09_15_48_48.0/ --caselist ../text_files/case_list_ADNI.txt --datapath dwi_diffsr.nii.gz --bvalpath dwi.bval --bvecpath dwi.bvec --cropsize 64 --output mrtrix_outputs_raw_diffsr_nopreprocc
