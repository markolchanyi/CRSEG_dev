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


# Declare an array of string with type
declare -a StringArray=("/autofs/space/nicc_003/users/olchanyi/data/HCP/100610"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/102311"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/102816"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/104416"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/105923"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/108323"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/109123"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/111514"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/114823"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/115017"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/115825"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/116726"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/118225"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/125525"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/126426"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/126931"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/128935"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/130518"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/131217"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/131722"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/132118"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/134627"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/134829"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/135124"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/137128"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/140117"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/144226"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/145834"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/146129"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/146432"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/146735"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/146937"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/148133"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/150423"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/155938"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/156334"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/157336"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/158035"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/158136"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/159239"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/162935"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/164131"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/164636"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/165436"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/167036"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/167440"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/169343"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/169747"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/171633"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/172130"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/173334"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/175237"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/176542"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/177140"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/177645"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/177746"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/178142"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/178243"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/178647"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/180533"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/181232"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/182436"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/182739"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/185442"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/186949"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/187345"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/191033"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/191336"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/191841"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/192439"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/192641"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/193845"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/195041"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/196144"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/197348"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/198653"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/199655"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/200210"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/200311"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/200614"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/201515"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/203418"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/204521"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/205220"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/209228"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/212419"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/214019"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/214524"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/221319"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/233326"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/239136"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/246133"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/249947"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/251833"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/257845"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/263436"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/283543"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/318637"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/320826"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/330324"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/346137"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/352738"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/360030"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/365343"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/380036"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/381038"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/389357"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/393247"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/395756"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/397760"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/406836"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/412528"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/429040"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/436845"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/463040"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/467351"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/525541"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/541943"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/547046"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/550439"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/562345"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/572045"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/573249"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/581450"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/601127"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/617748"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/627549"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/638049"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/654552"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/671855"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/680957"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/690152"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/706040"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/724446"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/725751"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/732243"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/745555"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/751550"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/757764"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/765864"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/770352"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/771354"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/782561"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/783462"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/789373"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/814649"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/818859"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/825048"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/826353"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/833249"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/859671"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/861456"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/871762"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/872764"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/878776"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/878877"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/898176"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/899885"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/901139"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/901442"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/905147"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/910241"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/926862"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/927359"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/942658"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/958976"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/966975"
                        "/autofs/space/nicc_003/users/olchanyi/data/HCP/995174")


for val in ${StringArray[@]}; do
        BASEPATH=$val
        #BASEPATH="/autofs/space/nicc_003/users/olchanyi/data/HCP/100610"
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


        ## run bedpostx
        if [ -e $BASEPATH/Native/dMRI/3T.bedpostX/merged_th1samples.nii.gz ]
        then
                echo "bedpost outputs already exist...skipping"
        else
                dwi2mask $datapath $PROCESSPATH/nodif_brain_mask.nii.gz -fslgrad $bvecpath $bvalpath -force
                echo "running bedpostx gpu"
                bedpostx_datacheck $PROCESSPATH
                sh /autofs/space/nicc_003/users/olchanyi/scripts/FS_scripts/bedpostx_helper_code.sh $PROCESSPATH
        fi




        # ----------- mrtrix BSB preprocessing script ----------- #
        if [ -e $BASEPATH/crseg_outputs/tracts_concatenated_1mm_cropped_norm.nii.gz ]
        then
                echo "Trackgen outputs already exist...skipping"
        else
                python ../CRSEG/trackgen.py \
                        --datapath $datapath \
                        --bvalpath $bvalpath \
                        --bvecpath $bvecpath \
                        --cropsize 64 \
                        --output $BASEPATH/crseg_outputs \
                        --use_fine_labels False
        fi




        # ----------- Unet WM segmentation script ----------- #
        if [ -e $BASEPATH/crseg_outputs/unet_predictions/unet_results/wmunet.seg.mgz ]
        then
                echo "Unet segmentation outputs already exist...skipping"
        else
                python ../CRSEG/unet_wm_predict.py \
                        --model_file /autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/joint_brainstem_model_v2/dice_090.h5 \
                        --output_path $BASEPATH/crseg_outputs/unet_predictions \
                        --lowb_file $BASEPATH/crseg_outputs/lowb_1mm_cropped_norm.nii.gz \
                        --fa_file $BASEPATH/crseg_outputs/fa_1mm_cropped_norm.nii.gz \
                        --tract_file $BASEPATH/crseg_outputs/tracts_concatenated_1mm_cropped_norm.nii.gz \
                        --label_list_path /autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/7ROI_training_dataset/brainstem_wm_label_list.npy
        fi




        if [ -e $BASEPATH/crseg_outputs/registration_outputs/AAN_label_volume_transformed.nii.gz ]
        then
                echo "CRSEG outputs already exist...skipping"
        else
                # ----------- CRSEG registration script ----------- #
                python ../scripts/CRSEG_main.py \
                        --target_fa_path $BASEPATH/crseg_outputs/fa_1mm_cropped.nii.gz \
                        --target_lowb_path $BASEPATH/crseg_outputs/lowb_1mm_cropped.nii.gz \
                        --wm_seg_path $BASEPATH/crseg_outputs/unet_predictions/unet_results/wmunet.seg.mgz \
                        --atlas_fa_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/FSL_HCP1065_FA_0.5mm_BRAINSTEM_CROPPED_128.nii.gz \
                        --atlas_lowb_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/T2_0.5mm_BRAINSTEM_CROPPED_128.nii.gz \
                        --atlas_aan_label_directory /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/AAN_probabilistic_labels_128 \
                        --label_list_path /autofs/space/nicc_003/users/olchanyi/Atlases/CRSEG_atlas/brainstem_wm_label_list.npy \
                        --atlas_wm_seg_path /autofs/space/nicc_003/users/olchanyi/scripts/ANTs/scripts/WMB_templates/CRSEG_ANTs_TEMPLATE.nii.gz \
                        --output_directory $BASEPATH/crseg_outputs/registration_outputs \
                        --resolution 1.0 \
                        --num_threads 1 \
                        --label_overlap 0.3
        fi




        if [ -e $BASEPATH/crseg_outputs/probtrackx_outputs/outputs/statistics.xlsx ]
        then
                echo "Probtrackx stats outputs already exist...skipping"
        else
                # ----------- run probtrackx script ----------- #
                python ../scripts/run_probtrackx.py \
                        --bedpost_path $BASEPATH/Native/dMRI/3T.bedpostX \
                        --seg_path $BASEPATH/crseg_outputs/registration_outputs/AAN_label_volume_transformed.nii.gz \
                        --probtrackx_path $BASEPATH/crseg_outputs/probtrackx_outputs \
                        --template_path $PROCESSPATH/nodif_brain_mask.nii.gz
        fi
done
