import glob
import os

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter as gauss_filt

import dtiutils
import utils


# This is the first generator I tried, deforming the FA linearly and the direction with nearest neighbors
# TODO: right now it takes 3.5 seconds on my machine... not great, not terrible
def image_seg_generator(training_dir,
                        path_label_list,
                        batchsize=1,
                        scaling_bounds=0.15,
                        rotation_bounds=15,
                        max_noise_std=0.1,
                        max_noise_std_fa=0.03,
                        gamma_std=0.1,
                        contrast_std=0.1,
                        brightness_std=0.1,
                        crop_size=None,
                        randomize_resolution=False,
                        diffusion_resolution=None):

    # Read directory to get list of training cases
    lowb_list = glob.glob(training_dir + '/dmri/lowb.nii.gz')
    n_training = len(lowb_list)
    print('Found %d cases for training' % n_training)

    # Get size from first volume
    aux, aff, _ = utils.load_volume(lowb_list[0], im_only=False)
    t1_resolution = np.sum(aff,axis=0)[:-1]
    nx, ny, nz = aux.shape
    if crop_size is None:
        crop_size = aux.shape
    if type(crop_size) == int:
        crop_size = [crop_size] *3

    # Create meshgrid (we will reuse a lot)
    xx, yy, zz = np.meshgrid(range(nx), range(ny), range(nz), sparse=False, indexing='ij')
    cx, cy, cz = (np.array(aux.shape) - 1) / 2
    xc = xx - cx
    yc = yy - cy
    zc = zz - cz

    # Some useful precomputations for one-hot encoding
    label_list = np.sort(np.load(path_label_list)).astype(int)
    mapping = np.zeros(1 + label_list[-1], dtype='int')
    mapping[label_list] = np.arange(len(label_list))

    xxcrop, yycrop, zzcrop = np.meshgrid(range(crop_size[0]), range(crop_size[1]), range(crop_size[2]), sparse=False, indexing='ij')
    xxcrop = xxcrop.flatten()
    yycrop = yycrop.flatten()
    zzcrop = zzcrop.flatten()



    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = np.random.randint(n_training, size=batchsize)

        # initialise input lists
        list_images = []
        list_label_maps = []

        for index in indices:

            # read images
            t1_file = lowb_list[index]
            prefix = lowb_list[index][:-12]
            high_prefix = lowb_list[index][:-16]
            fa_file = prefix + 'FA.nii.gz'
            v1_file = prefix + 'tracks.nii.gz'
            seg_file = high_prefix + 'seg.nii.gz'

            t1, aff, _ = utils.load_volume(t1_file, im_only=False)
            fa = utils.load_volume(fa_file)
            v1 = utils.load_volume(v1_file)
            seg = utils.load_volume(seg_file)

            # Sample augmentation parameters
            rotations = (2 * rotation_bounds * np.random.rand(3) - rotation_bounds) / 180.0 * np.pi
            s = 1 + (2 * scaling_bounds * np.random.rand(1) - scaling_bounds)
            cropx = np.random.randint(0, nx - crop_size[0] + 1, 1)[0]
            cropy = np.random.randint(0, ny - crop_size[1] + 1, 1)[0]
            cropz = np.random.randint(0, nz - crop_size[2] + 1, 1)[0]

            # Create random rotation matrix and scaling, and apply to coordinates,
            R = utils.make_rotation_matrix(rotations)
            Rinv = np.linalg.inv(R)

            xx2 = cx + s * (R[0, 0] * xc + R[0, 1] * yc + R[0, 2] * zc)
            yy2 = cy + s * (R[1, 0] * xc + R[1, 1] * yc + R[1, 2] * zc)
            zz2 = cz + s * (R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc)

            # Interpolate!  There is no need to interpolate everywhere; only in the area we will (randomly) crop
            # Essentially, we crop and interpolate at the same time
            xx2 = xx2[cropx:cropx+crop_size[0], cropy:cropy+crop_size[1], cropz:cropz+crop_size[2]]
            yy2 = yy2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]
            zz2 = zz2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]

            # We do FA and T1 in one shot with complex numbers :-)
            combo = np.array(t1, dtype=complex)
            combo.imag = fa
            combo_interpolator = rgi((range(nx), range(ny), range(nz)), combo, method='linear', bounds_error=False, fill_value=0.0)
            combo_def = combo_interpolator((xx2, yy2, zz2))
            t1_def = np.real(combo_def)
            fa_def = np.imag(combo_def)

            # We do the nearest neighbor interpolation ourselves
            idx, ok = utils.nn_interpolator_indices(xx2, yy2, zz2, nx, ny, nz)
            v1_def = np.zeros((*crop_size, 3))
            # TODO: the -1 in the first coordinate (left-right) -1 is crucial
            # I wonder if (/hope!) it's the same for every FreeSurfer processed dataset
            v1_def[:, :, :, 0][ok] = - v1[:, :, :, 0].flatten(order='F')[idx]
            v1_def[:, :, :, 1][ok] = v1[:, :, :, 1].flatten(order='F')[idx]
            v1_def[:, :, :, 2][ok] = v1[:, :, :, 2].flatten(order='F')[idx]
            seg_def = np.zeros(crop_size)
            seg_def[ok] = seg.flatten(order='F')[idx]

            # We also need to rotate v1
            v1_def_rot = np.zeros_like(v1_def)
            for row in range(3):
                for col in range(3):
                    v1_def_rot[:, :, :, row] = v1_def_rot[:, :, :, row] + Rinv[row, col] * v1_def[:, :, :, col]

            # Randomization of resolution increases running time by 0.5 seconds, which is not terrible...
            if randomize_resolution:
                # Random resolution for diffusion: between ~ 1 and 3 mm in each axis (but not too far from each other)
                aux = 1 + 2 * np.random.rand(1)
                batch_resolution_diffusion = aux + 0.2 * np.random.randn(3)
                batch_resolution_diffusion[batch_resolution_diffusion < 1] = 1 # let's be realistic :-)

                # Random resolution for t1: between 0.7 and 1.3 mm in each axis (but not too far from each other)
                aux = 0.7 + 0.6 * np.random.rand(1)
                batch_resolution_t1 = aux + 0.05 * np.random.randn(3)
                batch_resolution_t1[batch_resolution_diffusion < 0.6] = 0.6 # let's be realistic :-)

                # The theoretical blurring sigma to blur the resolution depends on the fraction by which we want to
                # divide the power at the cutoff frequency. I use [3, 20] which translates into multiplying the ratio
                # of resolutions by [0.35,0.95]
                fraction = 0.35 + 0.6 * np.random.rand(1)

                ratio_t1 = batch_resolution_t1 / t1_resolution
                ratio_t1[ratio_t1 < 1] = 1
                sigmas_t1 = fraction * ratio_t1
                sigmas_t1[ratio_t1 == 1] = 0 # Don't blur if we're not really going down in resolution

                ratio_diffusion = batch_resolution_diffusion / diffusion_resolution
                ratio_diffusion[ratio_diffusion < 1] = 1
                sigmas_diffusion = fraction * ratio_diffusion
                sigmas_diffusion[ratio_diffusion == 1] = 0 # Don't blur if we're not really going down in resolution

                # Low-pass filtering to blur data! Here we just blur the FA
                t1_def = gauss_filt(t1_def, sigmas_t1, truncate=3.0)
                fa_def = gauss_filt(fa_def, sigmas_diffusion, truncate=3.0)

                # Subsample: will require resampling / interpolating (sigh)
                t1_def = utils.subsample(t1_def, ratio_t1, crop_size, method='linear')
                fa_def, xig, yig, zig = utils.subsample(fa_def, ratio_diffusion, crop_size, method='linear', return_locations=True)
                # Nearest neighbor part for v1: we know all indices wil be inbounds
                idx = utils.nn_interpolator_indices_nocheck(xig, yig, zig, v1_def_rot.shape[0], v1_def_rot.shape[1], v1_def_rot.shape[2])
                v1_def_rot_downsampled = np.zeros([*fa_def.shape, 3])
                for c in range(3):
                    v1_def_rot_downsampled[:,:,:,c] = v1_def_rot[:, :, :, c].flatten(order='F')[idx]
                v1_def_rot = v1_def_rot_downsampled

            # Augment intensities t1 and fa, and compute  DTI (RGB) volume with new FA
            # Note that if you are downsampling, augmentation happens here at low resolution (as will happen at test time)
            fa_def = utils.augment_fa(fa_def, gamma_std, max_noise_std_fa)
            dti_def = np.abs(v1_def_rot * fa_def[..., np.newaxis])
            t1_def = utils.augment_t1(t1_def, gamma_std, contrast_std, brightness_std, max_noise_std)


            # Bring back to original resolution if needed
            if randomize_resolution:
                # TODO: it's crucial to upsample the same way as we do when predicting...

                # First the T1
                t1_def = utils.upsample(t1_def, ratio_t1, crop_size)
                # Now the diffusion
                fa_def, xig, yig, zig = utils.upsample(fa_def, ratio_diffusion, crop_size, return_locations=True)
                # again, careful with the eigenvectors...
                idx = utils.nn_interpolator_indices_nocheck(xig, yig, zig, v1_def_rot.shape[0], v1_def_rot.shape[1], v1_def_rot.shape[2])
                v1_def_rot_upsampled = np.zeros([*fa_def.shape, 3])
                for c in range(3):
                    v1_def_rot_upsampled[:,:,:,c] = v1_def_rot[:, :, :, c].flatten(order='F')[idx]
                v1_def_rot = v1_def_rot_upsampled

                dti_def = np.abs(v1_def_rot * fa_def[..., np.newaxis])


            # TODO: possible improvement: introduce left right flipping. You need to a) flip all the volumes, b) swap
            # left and right labels in the flipped segmentation, c) change the sign of the flipped v1_def_rot[:, :, :, 0]


            # Efficiently turn label map into one hot encoded array
            seg_def = mapping[seg_def.astype(int)]
            aux = np.zeros(t1_def.size * len(label_list))
            idx = xxcrop + yycrop * t1_def.shape[0] + zzcrop * t1_def.shape[0] * t1_def.shape[1] \
                  + seg_def.flatten() * t1_def.size # This is essentially a Matlab sub2ind
            aux[idx] = 1.0
            onehot = aux.reshape((*t1_def.shape, len(label_list)), order='F')

            # If you want to save to disk and open with Freeview during debugging
            # from joint_diffusion_structural_seg.utils import save_volume
            # utils.save_volume(t1, aff, None, '/tmp/t1.mgz')
            # utils.save_volume(t1_def, aff, None, '/tmp/t1_def.mgz')
            # utils.save_volume(fa, aff, None, '/tmp/fa.mgz')
            # utils.save_volume(fa_def, aff, None, '/tmp/fa_def.mgz')
            # utils.save_volume(seg, aff, None, '/tmp/seg.mgz')
            # utils.save_volume(seg_def, aff, None, '/tmp/seg_def.mgz')
            # utils.save_volume(v1, aff, None, '/tmp/v1.mgz')
            # utils.save_volume(v1_def_rot, aff, None, '/tmp/v1_def.mgz')
            # dti = np.abs(v1 * fa[..., np.newaxis])
            # utils.save_volume(dti * 255, aff, None, '/tmp/dti.mgz')
            # utils.save_volume(dti_def * 255, aff, None, '/tmp/dti_def.mgz')
            # utils.save_volume(onehot, aff, None, '/tmp/onehot_def.mgz')

            list_images.append(np.concatenate((t1_def[..., np.newaxis], fa_def[..., np.newaxis], dti_def), axis=-1)[np.newaxis,...])
            list_label_maps.append(onehot[np.newaxis,...])

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs



# This is a second attempt, working linearly on RGB space, which reduces the "Lego blocks" created by nearest neighbor interpolation
# Afer porting this to torch (rather than using numpy), an iteration with this generator takes about 1.5 seconds on my machine (much better than 7 seconds with numpy)
def image_seg_generator_rgb(training_dir,
                            path_label_list,
                            path_group_list,
                            batchsize=1,
                            scaling_bounds=0.15,
                            rotation_bounds=15,
                            nonlinear_rotation=False,
                            max_noise_std=0.1,
                            max_noise_std_fa=0.03,
                            gamma_std=0.1,
                            contrast_std=0.1,
                            brightness_std=0.1,
                            crop_size=None,
                            randomize_resolution=False,
                            diffusion_resolution=None,
                            speckle_frac_selected=1e-4,
                            randomize_flip=True,
                            seg_selection='grouped',
                            flag_deformation=True,
                            deformation_max = 5.0):

    # check type of one-hot encoding
    assert (seg_selection == 'single') or (seg_selection == 'combined') \
            or (seg_selection == 'mode') or (seg_selection == 'grouped') ,\
            'seg_selection must be single, combined, mode or grouped'

    # check speckle fraction is in range
    assert (speckle_frac_selected<1) and (speckle_frac_selected>=0),\
        'fraction of DTI voxels randomised must be between 0 and 1'

    if seg_selection =='grouped':
        grp_list = np.load(path_group_list)
        grp_mat = torch.zeros(grp_list.shape[0], grp_list.max() + 1, dtype=torch.float64)
        for il in range(0, grp_list.shape[0]):
            grp_mat[il, grp_list[il]] = 1

    else :
        grp_mat = None

    # only perform speckle when selection fraction is greater than 0
    randomize_speckle = speckle_frac_selected>0

    # Read directory to get list of training cases
    lowb_list = glob.glob(training_dir + '/subject*/dmri/lowb.nii.gz')
    n_training = len(lowb_list)
    print('Found %d cases for training' % n_training)

    # Get size from first volume
    aux, aff, _ = utils.load_volume(lowb_list[0], im_only=False)
    lowb_resolution = np.sum(aff,axis=0)[:-1]
    nx, ny, nz = aux.shape
    if crop_size is None:
        crop_size = aux.shape
    if type(crop_size) == int:
        crop_size = [crop_size] *3

    # Create meshgrid (we will reuse a lot)
    xx, yy, zz = np.meshgrid(range(nx+1), range(ny+1), range(nz+1), sparse=False, indexing='ij')
    cx, cy, cz = (np.array(aux.shape) - 1) / 2
    xc = xx - cx
    yc = yy - cy
    zc = zz - cz
    xc = torch.tensor(xc, device='cpu')
    yc = torch.tensor(yc, device='cpu')
    zc = torch.tensor(zc, device='cpu')
    cx = torch.tensor(cx, device='cpu')
    cy = torch.tensor(cy, device='cpu')
    cz = torch.tensor(cz, device='cpu')

    # Some useful precomputations for one-hot encoding
    label_list = np.sort(np.load(path_label_list)).astype(int)
    mapping = np.zeros(1 + label_list[-1], dtype='int')
    mapping[label_list] = np.arange(len(label_list))
    mapping = torch.tensor(mapping, device='cpu').long()



    # indices =0

    # Generate!
    # count = 0
    while True:

        # randomly pick as many images as batchsize
        indices = np.random.randint(n_training, size=batchsize)
        # indices = [count]

        # at end of loop indices += 1
        # overflow to 0 once through all examples

        # initialise input lists
        list_images = []
        list_label_maps = []

        for index in indices:

            # read images
            # TODO: this may go wrong with a larger batchsize
            lowb_file = lowb_list[index]
            subject_path = os.path.split(os.path.split(lowb_file)[0])[0]

            seg_list = glob.glob(subject_path + '/segs/*nii.gz')

            # either pick a single seg to train towards or import them all and average the onehot
            if seg_selection == 'single':
                seg_index = np.random.randint(len(seg_list))
                seg_file = seg_list[seg_index]
                seg = utils.load_volume(seg_file)
                seg = torch.tensor(seg, device='cpu').long()
            else:
                seg = utils.load_volume(seg_list[0])
                seg = torch.tensor(seg, device='cpu').long()
                seg = seg[..., None]
                for il in range(1, len(seg_list)):
                    np_seg = utils.load_volume(seg_list[il])
                    seg = torch.concat((seg, torch.tensor(np_seg[..., None], device='cpu')), dim=3)


            fa_list = glob.glob(subject_path + '/dmri/FA.nii.gz')
            fa_index = np.random.randint(len(fa_list))

            #fa_file = fa_list[fa_index]
            fa_file = fa_list[0]
            prefix = fa_file[:-10]
            v1_file = glob.glob(subject_path + '/dmri/tracts.nii.gz')[0]

            lowb, aff, _ = utils.load_volume(lowb_file, im_only=False)
            fa = utils.load_volume(fa_file)
            v1 = utils.load_volume(v1_file)
            lowb = torch.tensor(lowb, device='cpu')
            aff = torch.tensor(aff, device='cpu')
            fa = torch.tensor(fa, device='cpu')
            v1 = torch.tensor(v1, device='cpu')

            # Sample augmentation parameters
            rotations = (2 * rotation_bounds * np.random.rand(3) - rotation_bounds) / 180.0 * np.pi
            s = torch.tensor(1 + (2 * scaling_bounds * np.random.rand(1) - scaling_bounds), device='cpu')
            cropx = np.random.randint(0, nx - crop_size[0] + 1, 1)[0]
            cropy = np.random.randint(0, ny - crop_size[1] + 1, 1)[0]
            cropz = np.random.randint(0, nz - crop_size[2] + 1, 1)[0]

            # Create random rotation matrix and scaling, and apply to v1 and to coordinates,
            R = utils.make_rotation_matrix(rotations)
            Rinv = np.linalg.inv(R)
            R = torch.tensor(R, device='cpu')
            Rinv = torch.tensor(Rinv, device='cpu')

            # TODO: the -1 in the first coordinate (left-right) -1 is crucial
            # I wonder if (/hope!) it's the same for every FreeSurfer processed dataset
            #v1[:, :, :, 0] = - v1[:, :, :, 0]

            if not flag_deformation:
                if not nonlinear_rotation:
                    v1_rot = torch.zeros(v1.shape, device='cpu')
                    for row in range(3):
                        for col in range(3):
                            v1_rot[:, :, :, row] = v1_rot[:, :, :, row] + Rinv[row, col] * v1[:, :, :, col]
                else:
                    v1_rot = rotate_vector(Rinv, nx, ny, nz, rotation_bounds, v1)

                xx2 = cx + s * (R[0, 0] * xc + R[0, 1] * yc + R[0, 2] * zc)
                yy2 = cy + s * (R[1, 0] * xc + R[1, 1] * yc + R[1, 2] * zc)
                zz2 = cz + s * (R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc)

                # We use the rotated v1 to create the RGB (DTI) volume and then forget about V1 / nearest neighbor interpolation
                dti = np.abs(v1_rot * fa[..., np.newaxis])

                # Interpolate!  There is no need to interpolate everywhere; only in the area we will (randomly) crop
                # Essentially, we crop and interpolate at the same time
                xx2 = xx2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]
                yy2 = yy2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]
                zz2 = zz2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]

                combo = torch.concat((t1[..., None], dti), dim=-1)
                combo_def = fast_3D_interp_torch(combo, xx2, yy2, zz2, 'linear')
                lowb_def = combo_def[:, :, :, 0]
                dti_def = combo_def[:, :, :, 1:]

            else:



                # This function generates the cropped x, y and z interpolation coordinates and resamples the correctly
                # reoriented dti in rgb space using preservation of principle direction
                dti_def, xx2, yy2, zz2 = dtiutils.randomly_resample_dti_PPD(v1, fa, R, s, xc, yc, zc, cx, cy, cz, crop_size,
                                                                        cropx, cropy, cropz,flag_deformation,
                                                                        deformation_max, lowb_resolution,
                                                                        nonlinear_rotation, rotation_bounds)

                lowb_def = fast_3D_interp_torch(lowb, xx2, yy2, zz2, 'linear')
                fa_def = fast_3D_interp_torch(fa, xx2, yy2, zz2, 'linear')
                dti_def = fast_3D_interp_torch(v1, xx2, yy2, zz2, 'linear')
                for c in range(3):
                    dti_def[:,:,:,c] = fast_3D_interp_torch(v1[:,:,:,c], xx2, yy2, zz2, 'linear')

            # interpolate the segmentation
            seg_def = fast_3D_interp_torch(seg, xx2, yy2, zz2, 'nearest')

            # Randomization of resolution increases running time by 0.5-1.0 seconds, which is not terrible...
            if randomize_resolution:
                # Random resolution for diffusion: between ~ 1 and 1.5 mm in each axis (but not too far from each other)
                aux = 1 + 0.2*np.random.rand(1)
                batch_resolution_diffusion = aux + 0.1 * np.random.randn(3)
                batch_resolution_diffusion[batch_resolution_diffusion < 1] = 1 # let's be realistic :-)

                # Random resolution for t1: between 0.7 and 1.3 mm in each axis (but not too far from each other)
                #### NOT FOR TRACTOGRAPHY -Mark
                #aux = 0.7 + 0.6 * np.random.rand(1)
                #batch_resolution_lowb = aux + 0.05 * np.random.randn(3)
                #batch_resolution_lowb[batch_resolution_diffusion < 0.6] = 0.6 # let's be realistic :-)
                batch_resolution_lowb = batch_resolution_diffusion
                # The theoretical blurring sigma to blur the resolution depends on the fraction by which we want to
                # divide the power at the cutoff frequency. I use [3, 20] which translates into multiplying the ratio
                # of resolutions by [0.35,0.95]
                fraction = 0.35 + 0.6 * np.random.rand(1)

                ratio_lowb = batch_resolution_lowb / lowb_resolution
                ratio_lowb[ratio_lowb < 1] = 1
                sigmas_lowb = fraction * ratio_lowb
                sigmas_lowb[ratio_lowb == 1] = 0 # Don't blur if we're not really going down in resolution

                ratio_diffusion = batch_resolution_diffusion / diffusion_resolution
                ratio_diffusion[ratio_diffusion < 1] = 1
                sigmas_diffusion = fraction * ratio_diffusion
                sigmas_diffusion[ratio_diffusion == 1] = 0 # Don't blur if we're not really going down in resolution

                # Low-pass filtering to blur data!
                lowb_def = torch.tensor(gauss_filt(lowb_def, sigmas_lowb, truncate=3.0), device='cpu')
                fa_def = torch.tensor(gauss_filt(fa_def, sigmas_lowb, truncate=3.0), device='cpu')
                for c in range(3):
                    dti_def[:,:,:,c] = torch.tensor(gauss_filt(dti_def[:,:,:,c], sigmas_lowb, truncate=3.0), device='cpu')

                # Subsample
                lowb_def = myzoom_torch(lowb_def, 1 / ratio_lowb)
                fa_def = myzoom_torch(fa_def, 1 / ratio_lowb)
                dti_def = myzoom_torch(dti_def, 1 / ratio_lowb)
                #for c in range(3):
                #    dti_def[:,:,:,c] = myzoom_torch(dti_def[:,:,:,c], 1 / ratio_lowb)



            # Augment intensities t1 and fa
            # Note that if you are downsampling, augmentation happens here at low resolution (as will happen at test time)
            lowb_def = utils.augment_t1(lowb_def, gamma_std, contrast_std, brightness_std, max_noise_std)
            fa_def = utils.augment_t1(fa_def, gamma_std, contrast_std, brightness_std, max_noise_std)
            #for c in range(3):
            #        dti_def[:,:,:,c] = utils.augment_t1(dti_def[:,:,:,c], gamma_std, contrast_std, brightness_std, max_noise_std)
            dti_def = augment_tract_channel_intensities(dti_def)
            #dti_def = utils.augment_t1(dti_def, gamma_std, contrast_std, brightness_std, max_noise_std)
            #if randomize_speckle:
            #    dti_def, fa_def = speckle_dti_and_fa(dti_def, gamma_std, max_noise_std_fa,
            #                                         speckle_frac_selected)
            #else:
            #    dti_def, fa_def = augment_dti_and_fa(dti_def, gamma_std, max_noise_std_fa)

            # Bring back to original resolution if needed
            if randomize_resolution:
                # TODO: it's crucial to upsample the same way as we do when predicting...

                # Using ratio_t1 and ratio_diffusion may give a size that is off 1 pixel due to rounding
                ratio = (torch.tensor(seg_def.shape[:3], device='cpu') / torch.tensor(lowb_def.shape, device='cpu')).detach().numpy()
                lowb_def = myzoom_torch(lowb_def, ratio)
                fa_def = myzoom_torch(fa_def, ratio)
                #ratio = (torch.tensor(seg_def.shape[:3], device='cpu') / torch.tensor(dti_def.shape[:-1], device='cpu')).detach().numpy()
                dti_def = myzoom_torch(dti_def, ratio)
                #fa_def = torch.sqrt(torch.sum(dti_def * dti_def, dim=-1))

            # Efficiently turn label map into one hot encoded array
            onehot = encode_onehot(mapping, seg_def, label_list, seg_selection, grp_mat)

            # introduce left right flipping. You need to
            # a) flip all the volumes,
            # b) swap left and right labels in the flipped segmentation,
            # c) change the sign of the flipped v1_def_rot[:, :, :, 0]
            #       (but the RGB takes the absolute, so we can skip this)
            if randomize_flip:
                test_flip = torch.rand(1)[0] > 0.5
                if test_flip:
                    lowb_def = torch.flip(lowb_def, [0])
                    fa_def = torch.flip(fa_def, [0])
                    for c in range(3):
                        dti_def[:,:,:,c] = torch.flip(dti_def[:,:,:,c], [0])
                    #dti_def = torch.flip(dti_def, [0])

                    flip_idx = torch.cat((torch.zeros(1, dtype=torch.long),
                                          torch.arange((len(label_list)+1)/2, len(label_list), dtype=torch.long),
                                          torch.arange(1, (len(label_list)+1)/2, dtype=torch.long)), dim=0)

                    onehot = torch.flip(onehot[..., flip_idx], [0])



            # If you want to save to disk and open with Freeview during debugging
            #from utils import save_volume
            #utils.save_volume(lowb, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/lowb.mgz')
            #utils.save_volume(lowb_def, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/lowb_def.mgz')
            #utils.save_volume(fa, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/fa.mgz')
            #utils.save_volume(fa_def, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/fa_def.mgz')
            #utils.save_volume(seg, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/seg.mgz')
            #utils.save_volume(seg_def, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/seg_def.mgz')
            #utils.save_volume(v1, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/v1.mgz')
            #dti = np.abs(v1 * fa[..., np.newaxis])
            #utils.save_volume(dti * 255, aff, None, '/tmp/dti.mgz')
            #utils.save_volume(dti_def, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/v1_def.mgz')
            #utils.save_volume(dti_def / (fa_def[..., None] + 1e-6), aff, None, '/tmp/v1_def.mgz')
            #utils.save_volume(onehot, aff, None, '/autofs/space/nicc_003/users/olchanyi/CRSEG_dev/unet_scripts/scripts/tmp/onehot_def.mgz')

            list_images.append((torch.concat((lowb_def[..., None], fa_def[..., None], dti_def), axis=-1)[None, ...]).detach().numpy())
            list_label_maps.append((onehot[None, ...]).detach().numpy())

            # count += 1
            # if count == len(number of examples in validation set):
            #     count = 0

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs



def image_seg_generator_rgb_validation(validation_dir,
                            path_label_list,
                            batchsize=1,
                            scaling_bounds=0.15,
                            rotation_bounds=15,
                            max_noise_std=0.1,
                            max_noise_std_fa=0.03,
                            gamma_std=0.1,
                            contrast_std=0.1,
                            brightness_std=0.1,
                            crop_size=None,
                            randomize_resolution=False,
                            diffusion_resolution=None,
                            seg_selection='combined'):
    # check type of one-hot encoding
    assert (seg_selection == 'single') or (seg_selection == 'combined'),\
        'seg_selection must be single or combined'

    if not seg_selection == 'grouped':
        print("No group mat available...setting to None")
        grp_mat = None

    # Read directory to get list of training cases
    lowb_list = glob.glob(validation_dir + '/subject*/dmri/lowb.nii.gz')
    n_validation = len(lowb_list)
    print('Found %d cases for validation' % n_validation)

    # Get size from first volume
    aux, aff, _ = utils.load_volume(lowb_list[0], im_only=False)
    nx, ny, nz = aux.shape
    if crop_size is None:
        crop_size = aux.shape
    if type(crop_size) == int:
        crop_size = [crop_size] * 3

    # Some useful precomputations for one-hot encoding
    label_list = np.sort(np.load(path_label_list)).astype(int)
    mapping = np.zeros(1 + label_list[-1], dtype='int')
    mapping[label_list] = np.arange(len(label_list))
    mapping = torch.tensor(mapping, device='cpu').long()

    # Generate!
    count = 0
    while True:
        # randomly pick as many images as batchsize
        indices = list(range(count, count+batchsize))
        count += batchsize

        if count >= n_validation:
            count = 0

        # ToDo handle overflow within the indices when batchsize does not divide n_training
        # mostly won't affect us as we're using batchsize 1

        # initialise input lists
        list_images = []
        list_label_maps = []

        for index in indices:

            # read images
            # TODO: this may go wrong with a larger batchsize

            lowb_file = lowb_list[index]
            subject_path = os.path.split(os.path.split(lowb_file)[0])[0]

            seg_list = glob.glob(subject_path + '/segs/*nii.gz')

            # either pick a single seg to train towards or import them all and average the onehot
            if seg_selection == 'single':
                seg_index = np.random.randint(len(seg_list))
                seg_file = seg_list[seg_index]
                seg = utils.load_volume(seg_file)
                seg = torch.tensor(seg, device='cpu').long()
            else:
                seg = utils.load_volume(seg_list[0])
                seg = torch.tensor(seg, device='cpu').long()
                seg = seg[..., None]
                for il in range(1, len(seg_list)):
                    np_seg = utils.load_volume(seg_list[il])
                    seg = torch.concat((seg, torch.tensor(np_seg[..., None], device='cpu')), dim=3)


            fa_list = glob.glob(subject_path + '/dmri/FA.nii.gz')
            fa_index = np.random.randint(len(fa_list))

            #fa_file = fa_list[fa_index]
            fa_file = fa_list[0]
            prefix = fa_file[:-10]
            v1_file = glob.glob(subject_path + '/dmri/tracts.nii.gz')[0]

            lowb, aff, _ = utils.load_volume(lowb_file, im_only=False)
            fa = utils.load_volume(fa_file)
            v1 = utils.load_volume(v1_file)
            lowb = torch.tensor(lowb, device='cpu')
            aff = torch.tensor(aff, device='cpu')
            fa = torch.tensor(fa, device='cpu')
            v1 = torch.tensor(v1, device='cpu')

            # Sample augmentation parameters
            cropx = int(np.floor(nx - crop_size[0])/2)
            cropy = int(np.floor(ny - crop_size[1])/2)
            cropz = int(np.floor(nz - crop_size[2])/2)

            v1_crop = v1[cropx:cropx+crop_size[0],
                     cropy:cropy+crop_size[1],
                     cropz:cropz+crop_size[2], :]

            lowb_crop = lowb[cropx:cropx+crop_size[0],
                     cropy:cropy+crop_size[1],
                     cropz:cropz+crop_size[2]]

            seg_crop = seg[cropx:cropx+crop_size[0],
                     cropy:cropy+crop_size[1],
                     cropz:cropz+crop_size[2]]

            fa_crop = fa[cropx:cropx+crop_size[0],
                     cropy:cropy+crop_size[1],
                     cropz:cropz+crop_size[2]]

            #dti_crop = np.abs(v1_crop * fa_crop[..., np.newaxis])

            onehot = encode_onehot(mapping, seg_crop, label_list, seg_selection, grp_mat)

            list_images.append((torch.concat((lowb_crop[..., None], fa_crop[..., None], v1_crop), dim=-1)[None, ...]).detach().numpy())
            list_label_maps.append((onehot[None, ...]).detach().numpy())


        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs


def fast_3D_interp_torch(X, II, JJ, KK, mode):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], device='cpu')
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[:,:,:,channel] = aux[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], device='cpu')
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, device='cpu')
            Yc[ok] = c.float()
            Y[:,:,:,channel] = Yc

        if Y.shape[3]==1:
            Y = Y[:,:,:,0]

    else:
        raise Exception('mode must be linear or nearest')

    return Y

def myzoom_torch(X, factor):

    if len(X.shape)==3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], device='cpu')
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], device='cpu')
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], device='cpu')

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = np.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], device='cpu')

    for channel in range(X.shape[3]):
        Xc = X[:,:,:,channel]

        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], device='cpu')
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] +  wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], device='cpu')
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    return Y

# noinspection PyTypeChecker
def speckle_dti_and_fa(dti_def, gamma_std, max_noise_std_fa, speckle_frac_selected=0.0001):

    # Add some truly random DTI voxels
    selector = torch.rand(*dti_def.shape[:3]) > (1-speckle_frac_selected)
    n_selected = torch.sum(selector)
    # Randomise direction in selection
    dti_def[selector, :] = torch.rand((n_selected, 3), dtype=dti_def.dtype)

    # Gamma augmentation of the fa
    fa_def = torch.sqrt(torch.sum(dti_def * dti_def, dim=-1))
    fa_aug = utils.augment_fa(fa_def, gamma_std, max_noise_std_fa)

    # Replace speckle voxel FAs with higher values (often higher for voxels where the DTI model fails)
    fa_aug[selector] = 0.5 + 0.5 * torch.rand(n_selected, dtype=fa_aug.dtype)

    factor = fa_aug / (1e-6 + fa_def)
    dti_def = dti_def * factor[..., None]

    return dti_def, fa_def


def augment_dti_and_fa(dti_def, gamma_std, max_noise_std_fa):
    fa_def = torch.sqrt(torch.sum(dti_def * dti_def, dim=-1))
    fa_aug = utils.augment_fa(fa_def, gamma_std, max_noise_std_fa)

    factor = fa_aug / (1e-6 + fa_def)
    dti_def = dti_def * factor[..., None]

    return dti_def, fa_def


def augment_tract_channel_intensities(dti):
    aug = np.random.rand()
    if aug < 0.06:
        mgn_rand = 1 + np.random.normal(loc=1.0, scale=5.0, size=(3,1))
        for c in range(3):
            dti[:,:,:,c] = dti[:,:,:,c]*mgn_rand[c]
        return dti
    else:
        return dti

def encode_onehot(mapping, seg_def, label_list, seg_selection, grp_mat):
    if seg_selection == 'single':
        seg_def = mapping[seg_def.long()]
        eye = np.eye(len(label_list))
        eye = torch.tensor(eye, device='cpu')
        onehot = eye[seg_def]
    elif seg_selection == 'combined':
        seg_def = mapping[seg_def.long()]
        seg_def_max = torch.max(seg_def, dim=-1)[0]
        eye = np.eye(len(label_list))
        eye = torch.tensor(eye, device='cpu')
        onehot = eye[seg_def_max]
        # use max onehot to get background right then replace the thalamus with averaged
        onehot[seg_def_max > 0, :] = torch.sum(eye[seg_def[seg_def_max > 0, :]], dim=-2)
        onehot /= torch.sum(onehot, dim=-1, keepdim=True)
    elif seg_selection == 'mode':
        seg_def = mapping[seg_def.long()]
        seg_def_mode = torch.mode(seg_def, dim=-1)[0]
        eye = np.eye(len(label_list))
        eye = torch.tensor(eye, device='cpu')
        onehot = eye[seg_def_mode]
    else :
        seg_def = mapping[seg_def.long()]
        seg_def_max = torch.max(seg_def, dim=-1)[0]
        eye = np.eye(len(label_list))
        eye = torch.tensor(eye, device='cpu')
        onehot = eye[seg_def_max]
        # use max onehot to get background right then replace the thalamus with averaged
        thal_average = torch.sum(eye[seg_def[seg_def_max > 0, :]], dim=-2)
        # summ over grouped labels
        soft_mask = thal_average @ grp_mat
        # keep only most likely group
        mask = (soft_mask == soft_mask.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float64) @ grp_mat.t()
        onehot[seg_def_max > 0, :] = thal_average * mask
        onehot /= torch.sum(onehot, dim=-1, keepdim=True)
        # y = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(y), self.group_seg, num_segments=self.n_groups ))



    return onehot

def gen_non_linear_rotations(seed_size, crop_size, rotation_sd, Rinv=None, device='cpu'):

    rot_seed = (2 * rotation_sd * torch.rand(seed_size[0], seed_size[1], seed_size[2], 3, device=device)) - rotation_sd

    sin_seed = torch.sin(rot_seed)
    cos_seed = torch.cos(rot_seed)

    Rx_seed = torch.zeros(seed_size[0], seed_size[1], seed_size[2], 3, 3,  device=device)
    Rx_seed[..., 0, 0] = 1
    Rx_seed[..., 1, 1] = cos_seed[..., 0]
    Rx_seed[..., 1, 2] = -sin_seed[..., 0]
    Rx_seed[..., 2, 1] = sin_seed[..., 0]
    Rx_seed[..., 2, 2] = cos_seed[..., 0]

    Ry_seed = torch.zeros(seed_size[0], seed_size[1], seed_size[2], 3, 3,  device=device)
    Ry_seed[..., 1, 1] = 1
    Ry_seed[..., 0, 0] = cos_seed[..., 1]
    Ry_seed[..., 2, 0] = -sin_seed[..., 1]
    Ry_seed[..., 0, 2] = sin_seed[..., 1]
    Ry_seed[..., 2, 2] = cos_seed[..., 1]

    Rz_seed = torch.zeros(seed_size[0], seed_size[1], seed_size[2], 3, 3,  device=device)
    Rz_seed[..., 2, 2] = 1
    Rz_seed[..., 0, 0] = cos_seed[..., 2]
    Rz_seed[..., 0, 1] = -sin_seed[..., 2]
    Rz_seed[..., 1, 0] = sin_seed[..., 2]
    Rz_seed[..., 1, 1] = cos_seed[..., 2]

    R_seed = torch.matmul(torch.matmul(Rx_seed, Ry_seed), Rz_seed)

    if Rinv is not None:
        R_seed = torch.matmul(R_seed, Rinv.detach().float())

    R_nonLin = torch.nn.functional.interpolate(torch.permute(R_seed, (3, 4, 0, 1, 2)),
                                               size=(crop_size[0], crop_size[1], crop_size[2]),
                                               mode='trilinear',
                                               align_corners=True)

    R_nonLin = torch.permute(R_nonLin, (2, 3, 4, 0, 1))

    return R_nonLin

def rotate_vector(Rinv, nx, ny, nz, rotation_bounds, v1):
    Rinv = gen_non_linear_rotations([5] * 3, [nx, ny, nz], (rotation_bounds / 360) * np.pi, Rinv=Rinv)
    v1_rot = torch.matmul(Rinv, v1[..., None])[:, :, :, :, 0]
    v1_rot /= (torch.sqrt(torch.sum(v1_rot * v1_rot, dim=-1, keepdim=True)) + 1e-6)
    return v1_rot

def interp_onehot(label_list, onehot_in, thal_mask, xx2, yy2, zz2, cx, cy, cz):

    idx, idy, idz = torch.nonzero(thal_mask>0,as_tuple=True)

    i1 = torch.min(idx)
    j1 = torch.min(idy)
    k1 = torch.min(idz)
    i2 = torch.max(idx)+1
    j2 = torch.max(idy)+1
    k2 = torch.max(idz)+1

    xx3 = (xx2[i1:i2, j1:j2, k1:k2, None]/cx)-1
    yy3 = (yy2[i1:i2, j1:j2, k1:k2, None]/cy)-1
    zz3 = (zz2[i1:i2, j1:j2, k1:k2, None]/cz)-1

    grid = torch.concat((zz3, yy3, xx3), dim=3)
    grid = grid[None, ...]
    grid = grid.detach().float()

    onehot_interp = torch.nn.functional.grid_sample(torch.permute(onehot_in, (3, 0, 1, 2))[None, ...],
                                                    grid, align_corners=True)

    eye = torch.eye(len(label_list))
    onehot_out = eye[torch.zeros_like(thal_mask, dtype=torch.long)]

    onehot_out[i1:i2, j1:j2, k1:k2, :] = torch.permute(onehot_interp[0, ...], (1, 2, 3, 0))

    return onehot_out
