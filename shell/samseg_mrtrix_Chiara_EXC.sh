#source freeurfer for samseg
export FREESURFER_HOME="/usr/local/freesurfer/7.2.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#run mrtrix preprocessing
python ../CRSEG/trackgen.py --basepath /space/nicc/3/users/cmaffei/EXC_connectom/ --caselist ../text_files/case_list_Chiara_EXC.txt --datapath dwi_eddy_oriented.nii.gz --bvalpath bval_c --bvecpath bvec_c_col --cropsize 90 --output mark_mrtrix_bsb_outputs
