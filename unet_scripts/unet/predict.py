import os

import numpy as np
from scipy import ndimage

import utils as utils
import models

def predict(subject_list,
                fs_subject_dir,
                dataset,
                path_label_list,
                model_file,
                resolution_model_file=1.0,
                generator_mode='rgb',
                unet_feat_count=24,
                n_levels=5,
                conv_size=3,
                feat_multiplier=2,
                nb_conv_per_level=2,
                activation='elu',
                bounding_box_width=64,
                aff_ref=np.eye(4),
                shell_flag=None):


    
    assert (generator_mode == 'fa_v1') | (generator_mode == 'rgb'), \
        'generator mode must be fa_v1 or rgb'

    assert dataset in ('HCP','ADNI','template','validate','DRC') #will do for now

    # Load label list
    label_list = np.load(path_label_list)

    # Build Unet
    unet_input_shape = [bounding_box_width, bounding_box_width, bounding_box_width, 5]
    n_labels = len(label_list)

    unet_model = models.unet(nb_features=unet_feat_count,
                             input_shape=unet_input_shape,
                             nb_levels=n_levels,
                             conv_size=conv_size,
                             nb_labels=n_labels,
                             feat_mult=feat_multiplier,
                             nb_conv_per_level=nb_conv_per_level,
                             conv_dropout=0,
                             batch_norm=-1,
                             activation=activation,
                             input_model=None)

    unet_model.load_weights(model_file, by_name=True)


    ### iteratre over subjects
    for subject in subject_list:

        if not os.path.exists(os.path.join(fs_subject_dir, subject, 'results')):
            os.mkdir(os.path.join(fs_subject_dir, subject, 'results'))
        output_seg_file = os.path.join(fs_subject_dir, subject, 'results', 'bsNet.seg.mgz')
        output_vol_file = os.path.join(fs_subject_dir, subject, 'results', 'bsNet.vol.npy')

        # File names
        if dataset=='HCP':
            t1_file = os.path.join(fs_subject_dir, subject, 'mri', 'T1w_hires.masked.norm.mgz')
            aseg_file = os.path.join(fs_subject_dir, subject, 'mri', 'aseg.mgz')
            fa_file = os.path.join(fs_subject_dir, subject, 'dmri', 'dtifit.1+2+3K_FA.nii.gz')
            v1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'dtifit.1+2+3K_V1.nii.gz')


        if dataset == 'template':
            t1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'lowb.nii.gz')
            fa_file = os.path.join(fs_subject_dir, subject, 'dmri', 'FA.nii.gz')
            v1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'tracts.nii.gz')


        # Read in and reorient T1
        t1, aff, _ = utils.load_volume(t1_file, im_only=False)
        print("SIZE OF T1 IS: ", t1.shape)
        t1, aff2 = utils.align_volume_to_ref(t1, aff, aff_ref=aff_ref, return_aff=True, n_dims=3)

        # If the resolution is not the one the model expected, we need to upsample!
        if any(abs(np.diag(aff2)[:-1] - resolution_model_file) > 0.1):
            print('Warning: t1 does not have the resolution that the CNN expects; we need to resample')
            t1, aff2 = utils.rescale_voxel_size(t1, aff2, [resolution_model_file, resolution_model_file, resolution_model_file])

        # Normalize the T1
        #wm_mask = (aseg == 2) | (aseg == 41)
        #wm_t1_median = np.median(t1[wm_mask])
        #t1 = t1 / wm_t1_median * 0.75
        #t1[t1 < 0] = 0
        #t1[t1 > 1] = 1

        # Find the center of the thalamus and crop a volumes around it
        #th_mask = (aseg == 10) | (aseg == 49)
        #idx = np.where(th_mask)
        #i1 = (np.mean(idx[0]) - np.round(0.5 * bounding_box_width)).astype(int)
        #j1 = (np.mean(idx[1]) - np.round(0.5 * bounding_box_width)).astype(int)
        #k1 = (np.mean(idx[2]) - np.round(0.5 * bounding_box_width)).astype(int)
        #i2 = i1 + bounding_box_width
        #j2 = j1 + bounding_box_width
        #k2 = k1 + bounding_box_width

        #t1 = t1[i1:i2, j1:j2, k1:k2]
        #aff2[:-1, -1] = aff2[:-1, -1] + np.matmul(aff2[:-1, :-1], np.array([i1, j1, k1])) # preserve the RAS coordinates

        # Now the diffusion data
        # We only resample in the cropped region
        # TODO: we'll want to do this in the log-tensor domain
        if generator_mode=='fa_v1':
            fa, aff, _ = utils.load_volume(fa_file, im_only=False)
            fa = utils.resample_like(t1, aff2, fa, aff)
            v1, aff, _ = utils.load_volume(v1_file, im_only=False)
            v1_copy = v1.copy()
            v1 = np.zeros([*t1.shape, 3])
            v1[:, :, :, 0] = - utils.resample_like(t1, aff2, v1_copy[:, :, :, 0], aff, method='nearest') # minus as in generators.py
            v1[:, :, :, 1] = utils.resample_like(t1, aff2, v1_copy[:, :, :, 1], aff, method='nearest')
            v1[:, :, :, 2] = utils.resample_like(t1, aff2, v1_copy[:, :, :, 2], aff, method='nearest')
            #dti = np.abs(v1 * fa[..., np.newaxis])
        else:
            fa, aff, _ = utils.load_volume(fa_file, im_only=False)
            v1 = utils.load_volume(v1_file, im_only=True)
            dti = np.abs(v1 * fa[..., np.newaxis])
            fa = utils.resample_like(t1, aff2, fa, aff)
            dti = utils.resample_like(t1, aff2, v1, aff)

        # Predict with left-right flipping augmentation
        input = np.concatenate((t1[..., np.newaxis], fa[..., np.newaxis], dti), axis=-1)[np.newaxis,...]
        posteriors = np.squeeze(unet_model.predict(input))
        posteriors_flipped = np.squeeze(unet_model.predict(input[:,::-1,:,:,:]))
        nlab = int(( len(label_list) - 1 ) / 2)
        posteriors[:,:,:,0] = 0.5 * posteriors[:,:,:,0] + 0.5 *  posteriors_flipped[::-1,:,:,0]
        posteriors[:,:,:,1:nlab+1] = 0.5 * posteriors[:,:,:,1:nlab+1] + 0.5 *  posteriors_flipped[::-1,:,:,nlab+1:]
        posteriors[:,:,:,nlab+1:] = 0.5 * posteriors[:,:,:,nlab+1:] + 0.5 *  posteriors_flipped[::-1,:,:,1:nlab+1]

        # Fill holes
        #thal_mask = posteriors[..., 0] < 0.5
        #thal_mask = ndimage.binary_fill_holes(thal_mask)


        # remove stray voxels with conn comps
        #thal_mask_copy = thal_mask.copy()
        # min_size: size of largest objects to remove
        # connectivity: for connected components, 1 is 6-connected, 2 is
        # 18-connected, and 3 is 26-connected   
        #min_size = 10
        #connectivity = 3
        #thal_mask = morphology.remove_small_objects(thal_mask_copy, min_size=min_size, connectivity=connectivity)
            
        ### OR just remove largest connected component
        #cc_labels = label(thal_mask_copy,connectivity=connectivity)
        #thal_mask = cc_labels == np.argmax(np.bincount(cc_labels.flat, weights=thal_mask_copy.flat))



        #posteriors[thal_mask, 0] = 0
        #posteriors /= np.sum(posteriors, axis=-1)[..., np.newaxis]

        #components, n_components = ndimage.label(thal_mask)

        # TODO: it'd be good to do some postprocessing here. I would do something like:
        # 1. Create a mask for the whole left thalamus, as the largest connected component of the union of left labels
        # 2. Dilate the mask eg by 3 voxels.
        # 3. Set to zero the probability of all left thalamic nuclei in the voxels outside the mask
        # 4. Repeat 1-3 with the right thalamus
        # 5. Renormalize posteriors by dividing by their sum (plus epsilon)

        # Compute volumes (skip background)
        voxel_vol_mm3 = np.linalg.det(aff2)
        vols_in_mm3 = np.sum(posteriors, axis=(0,1,2))[1:] * voxel_vol_mm3

        # Compute segmentations
        seg = label_list[np.argmax(posteriors, axis=-1)]

        # Write to disk and we're done!
        utils.save_volume(seg.astype(int), aff2, None, output_seg_file)
        np.save(output_vol_file, vols_in_mm3)

        print('freeview ' + t1_file + ' ' + fa_file + ' '  + output_seg_file)

        print('All done!')
