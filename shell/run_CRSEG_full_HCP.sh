#source freeurfer for samseg
#7.3.0 is the most stable version that does not throw a BLAS error
#and can also run synthSR and brainstem_subfield seg
export FREESURFER_HOME="/usr/local/freesurfer/7.3.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

set -e # The script will terminate after the first line that fails



BASEPATH="/autofs/space/nicc_003/users/olchanyi/data/HCP100/100307"
echo basepath provided is: $BASEPATH

# ----------- mrtrix BSB preprocessing script ----------- #
python ../CRSEG/trackgen.py \
        --datapath $BASEPATH/T1w/Diffusion/data.nii.gz \
        --bvalpath $BASEPATH/T1w/Diffusion/bvals \
        --bvecpath $BASEPATH/T1w/Diffusion/bvecs \
        --cropsize 64 \
        --output $BASEPATH/crseg_outputs \
        --use_fine_labels False \




# ----------- Unet WM segmentation script ----------- #
python ../CRSEG/unet_wm_predict.py \
        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/joint_brainstem_model_v2/dice_090.h5 \
        --output_path $BASEPATH/crseg_outputs/unet_predictions \
        --lowb_file $BASEPATH/crseg_outputs/lowb_1mm_cropped_norm.nii.gz \
        --fa_file $BASEPATH/crseg_outputs/fa_1mm_cropped_norm.nii.gz \
        --tract_file $BASEPATH/crseg_outputs/tracts_concatenated_1mm_cropped_norm.nii.gz \
        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/7ROI_training_dataset/brainstem_wm_label_list.npy \




# ----------- CRSEG registration script ----------- #
python ../scripts/CRSEG_main.py \
        --target_fa_path /autofs/space/nicc_003/users/olchanyi/scratch/subject_115320/dmri/FA.nii.gz \
        --target_lowb_path /autofs/space/nicc_003/users/olchanyi/scratch/subject_115320/dmri/lowb.nii.gz \
        --wm_seg_path /autofs/space/nicc_003/users/olchanyi/scratch/subject_115320/segs/seg.nii.gz \
        --atlas_fa_path ../../../Atlases/CRSEG_atlas/FSL_HCP1065_FA_0.5mm_BRAINSTEM_CROPPED.mgz \
        --atlas_lowb_path ../../../Atlases/CRSEG_atlas/T2_0.5mm_BRAINSTEM_CROPPED.mgz \
        --atlas_aan_label_directory /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/AAN_probabilistic_labels \
        --label_list_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/brainstem_wm_label_list.npy \
        --atlas_wm_seg_path ../../../Atlases/CRSEG_atlas/CRSEG_ROIs.mgz \
        --output_directory /autofs/space/nicc_003/users/olchanyi/scratch/subject_115320/CRSEG_outputs/ \
        --resolution 1.25 \
        --num_threads 1 \
        --label_overlap 0.3 \
