import numpy as np 
import sys

sys.path.append('/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/unet')

from predict import predict


# keep in list form since you can iterate over multiple sunjects explicitely...implicit will be added soon
subject_list = ['subject_115320' ,'subject_133019','subject_178950','subject_189450','subject_654754']
fs_subject_dir = '/autofs/space/nicc_003/users/olchanyi/data/CRSEG_unet_training_data/test_data/'
# for now...must be
dataset = 'template'
path_label_list = '../../../data/CRSEG_unet_training_data/7ROI_training_dataset/brainstem_wm_label_list.npy'
model_file = '/autofs/space/nicc_003/users/olchanyi/models/CRSEG_unet_models/joint_brainstem_model_v2/dice_090.h5'
# model file resolution
resolution_model_file=1.0
# generator mode for prediction data (make sure same as training!)
generator_mode='rgb'
# U-net: number of features in base level (make sure same as training!)
unet_feat_count=24
# U-net: number of levels (make sure same as training!)
n_levels = 5
# U-net: convolution kernel size (make sure same as training!)
conv_size = 3
# U-net: number of features per level multiplier (make sure same as training!)
feat_multiplier = 2
# U-net: number of convolutions per level (make sure same as training!)
nb_conv_per_level = 2
# U-net: activation function (make sure same as training!)
activation='elu'
# (isotropic) dimensions of bounding box to take around thalamus
bounding_box_width = 64
# reference affine
aff_ref = np.eye(4)


predict(subject_list,
            fs_subject_dir,
            dataset,
            path_label_list,
            model_file,
            resolution_model_file,
            generator_mode,
            unet_feat_count,
            n_levels,
            conv_size,
            feat_multiplier,
            nb_conv_per_level,
            activation,
            bounding_box_width,
            aff_ref)
