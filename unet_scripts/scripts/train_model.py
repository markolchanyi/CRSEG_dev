import os

from joint_diffusion_structural_seg.training import train

## machine specific directories
top_level_training_dir = '/media/henry/_localstore/Brain/synthDTI/large_download/training_reduced'
top_level_model_dir = '/media/henry/_localstore/Brain/synthDTI/models'

## Run specific parameters that will change for ablations
# Name of the model - link to ablation spreadsheet
model_name = 'joint_thalamus_test_Jan'
# Fraction of DTI voxels to randomised. Between 0 and 1. Set to 0 to turn off speckle. 1 in 10k sounds right
speckle_frac_selected=1e-4
# Flag whether we'll individually rotate the DTI vectors
nonlinear_rotation=True
# Will we deform the images with piecewise linear displacement fields (this includes re-orientation)
flag_deformation = True
# Maximimum piecewise linear displacement in mm (excluding rotation + scaling)
deformation_max = 5.0
# How to encode the segmentations into a onehot
# single (pick an example), combined (average), mode (majority vote), grouped (majority vote on group then average)
seg_selection='grouped'
# Dice version - "individual" do standard label-wise Dice, "grouped" also add contribution of groups and whole thalamus
dice_version="grouped"


## Paths
# Path with training data
training_dir = os.path.join(top_level_training_dir,'train/')
# Path with Validation data, set to none if not doing online validation
#validation_dir = os.path.join(top_level_training_dir,'validate/')
validation_dir = None
# NPY file with list of labels
path_label_list = os.path.join(top_level_training_dir,'proc_training_data_label_list_reduced.npy')
# NPY file with segmentation of onehot channels into groups for mixed Dice etc.
path_group_list = os.path.join(top_level_training_dir,'proc_training_group_seg_reduced.npy')
# Directory where model files will be written
model_dir = os.path.join(top_level_model_dir,model_name)

## Generator specific parameters that remain constant between runs
# Maximum scaling during augmentation, in [0,1]. 0.15 is a good value
scaling_bounds = 0.15
# Maximum rotation during augmentation, in degrees. 15 is a good value
rotation_bounds = 15
# Maximum standard deviation of noise to add in augmentation.
# Since images are normalized, 0.1 is a good value for t1 (a bit lower for FA, since it's a fit already)
max_noise_std = 0.1
max_noise_std_fa = 0.03
# Standard deviation of log gamma, both for FA and intensity channels
gamma_std = 0.1
# Standard deviation of random contrast / brightness for intensity channels
contrast_std = 0.1
brightness_std = 0.1
# Randomize resolution during training?
randomize_resolution = True
# Mode of the generator. Must be fa_v1 (linear interpolation of fa, nearest of v1) or rgb (linear on rgb)
generator_mode = 'rgb'
# Resolution of diffusion data (only needed if randomizing resolution; we use it to compute width of blurring kernels)
diffusion_resolution = 1.25


## Network specific parameters
# Batch size being volumes, it will probably be always 1...
batchsize = 1
# Size to which inputs will be cropped (use None to use whole volume)
crop_size = 128
# Number of levels in Unet (5 is good)
n_levels = 5
# Number of convolution + nonlinearity blocks per level (2 is good)
nb_conv_per_level = 2
# Size of convoluton kernels (typically 3)
conv_size = 3
# Number of features per layer (eg 24)
unet_feat_count = 24
# Feature multiplier, to have more features deeper in the net. We used to do 2, more recently Benjamin started using 1
feat_multiplier = 2
# Dropout probability (between 0 and 1, we normally disable it by setting it to 0)
dropout = 0
# Type of activation / nonlinearity (elu is good)
activation = 'elu'
# Learning rate: 1e-3 is too muchn, 1e-5 is generally too little, so 1e-4 is good
lr = 1e-4
# Decay in learning rate, if you want to schedule. I normally leave it alone (ie set it to 0)
lr_decay = 0
# Number of "pretraining" epochs where we use the L2 norm on the activations rather than Dice in the softmax (5-10)
wl2_epochs = 5
# Number of epocs with Dice
dice_epochs = 200
# Steps per epoch (1000 is good)
steps_per_epoch = 1000
# Checkpoint file from which training will start (use None to start from scratch)
checkpoint = None
# frequency of saving model checkpoints (Dice iterations only)
checkpoint_frequency = 5

train(training_dir,
             path_label_list,
             model_dir,
             path_group_list=path_group_list,
             validation_dir=validation_dir,
             batchsize=batchsize,
             crop_size=crop_size,
             scaling_bounds=scaling_bounds,
             rotation_bounds=rotation_bounds,
             nonlinear_rotation=nonlinear_rotation,
             max_noise_std=max_noise_std,
             max_noise_std_fa=max_noise_std_fa,
             gamma_std=gamma_std,
             contrast_std=contrast_std,
             brightness_std=brightness_std,
             randomize_resolution=randomize_resolution,
             generator_mode=generator_mode,
             diffusion_resolution=diffusion_resolution,
             speckle_frac_selected=speckle_frac_selected,
             seg_selection=seg_selection,
             flag_deformation=flag_deformation,
             deformation_max=deformation_max,
             n_levels=n_levels,
             nb_conv_per_level=nb_conv_per_level,
             conv_size=conv_size,
             unet_feat_count=unet_feat_count,
             feat_multiplier=feat_multiplier,
             dropout=dropout,
             activation=activation,
             lr=lr,
             lr_decay=lr_decay,
             wl2_epochs=wl2_epochs,
             dice_epochs=dice_epochs,
             steps_per_epoch=steps_per_epoch,
             checkpoint=checkpoint,
             dice_version=dice_version,
             checkpoint_frequency=checkpoint_frequency)


