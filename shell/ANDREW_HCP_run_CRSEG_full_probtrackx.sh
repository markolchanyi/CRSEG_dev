#source freeurfer for samseg
#7.3.0 is the most stable version that does not throw a BLAS error
#and can also run synthSR and brainstem_subfield seg
export FREESURFER_HOME=/usr/local/freesurfer/7.3.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
source /usr/pubsw/packages/mrtrix/env.sh
mrtrixdir=/usr/pubsw/packages/mrtrix/3.0.2/bin
fsldir=/usr/pubsw/packages/fsl/6.0.3/bin
FSLDIR=/usr/pubsw/packages/fsl/6.0.3
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/9.1/lib64
#export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/11.6/lib64:/usr/pubsw/packages/CUDA/11.6/extras/CUPTI/lib64:/usr/pubsw/packages/CUDA/9.0/lib64:/usr/pubsw/packages/CUDA/9.1/lib64

set -e # The script will terminate after the first line that fails




BASEPATH="/autofs/space/nicc_003/users/olchanyi/data/HCP/100610"
echo basepath provided is: $BASEPATH

## extract brain mask
datapath=$BASEPATH/Native/dMRI/3T/data.nii.gz
bvalpath=$BASEPATH/Native/dMRI/3T/bvals
bvecpath=$BASEPATH/Native/dMRI/3T/bvecs
PROCESSPATH=$BASEPATH/Native/dMRI/3T


## extract brain mask
#if [ ! -d "$PROCESSPATH/diff" ]; then
#  echo "$PROCESSPATH/diff does not exist...creating"
#  mkdir $PROCESSPATH/diff
#fi
dwi2mask $datapath $PROCESSPATH/nodif_brain_mask.nii.gz -fslgrad $bvecpath $bvalpath -force


## run bedpostx
if [ -e $BASEPATH/Native/dMRI/3T.bedpostX/merged_th1samples.nii.gz ]
then
        echo "bedpost outputs already exist...skipping"
else
        echo "running bedpostx gpu"
        bedpostx_datacheck $PROCESSPATH
        sh /autofs/space/nicc_003/users/olchanyi/scripts/FS_scripts/bedpostx_helper_code.sh $PROCESSPATH
fi



# ----------- mrtrix BSB preprocessing script ----------- #
#python ../CRSEG/trackgen.py \
#        --datapath $datapath \
#        --bvalpath $bvalpath \
#        --bvecpath $bvecpath \
#        --cropsize 64 \
#        --output $BASEPATH/crseg_outputs \
#        --use_fine_labels False \



# ----------- Unet WM segmentation script ----------- #
#python ../CRSEG/unet_wm_predict.py \
#        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/joint_brainstem_model_v2/dice_090.h5 \
#        --output_path $BASEPATH/crseg_outputs/unet_predictions \
#        --lowb_file $BASEPATH/crseg_outputs/lowb_1mm_cropped_norm.nii.gz \
#        --fa_file $BASEPATH/crseg_outputs/fa_1mm_cropped_norm.nii.gz \
#        --tract_file $BASEPATH/crseg_outputs/tracts_concatenated_1mm_cropped_norm.nii.gz \
#        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/7ROI_training_dataset/brainstem_wm_label_list.npy \



# ----------- CRSEG registration script ----------- #
#python ../scripts/CRSEG_main.py \
#        --target_fa_path $BASEPATH/crseg_outputs/fa_1mm_cropped.nii.gz \
#        --target_lowb_path $BASEPATH/crseg_outputs/lowb_1mm_cropped.nii.gz \
#        --wm_seg_path $BASEPATH/crseg_outputs/unet_predictions/unet_results/wmunet.seg.mgz \
#        --atlas_fa_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/FSL_HCP1065_FA_0.5mm_BRAINSTEM_CROPPED_128.nii.gz \
#        --atlas_lowb_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/T2_0.5mm_BRAINSTEM_CROPPED_128.nii.gz \
#        --atlas_aan_label_directory /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/AAN_probabilistic_labels_128 \
#        --label_list_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/brainstem_wm_label_list.npy \
#        --atlas_wm_seg_path /autofs/space/nicc_003/users/olchanyi/scripts/ANTs/scripts/WMB_templates/CRSEG_ANTs_TEMPLATE.nii.gz \
#        --output_directory $BASEPATH/crseg_outputs/registration_outputs \
#        --resolution 1.0 \
#        --num_threads 1 \
#        --label_overlap 0.3 \



# ----------- run probtrackx script ----------- #
python ../scripts/run_probtrackx.py \
        --bedpost_path $BASEPATH/Native/dMRI/3T.bedpostX \
        --seg_path $BASEPATH/crseg_outputs/registration_outputs/AAN_label_volume_transformed.nii.gz \
        --probtrackx_path $BASEPATH/crseg_outputs/probtrackx_outputs \
        --template_path $PROCESSPATH/nodif_brain_mask.nii.gz \
