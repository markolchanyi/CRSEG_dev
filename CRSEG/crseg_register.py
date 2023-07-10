import sys
import shutil
import os
import re
import math
import numpy as np
import torch as th
import nibabel as nib
import nibabel.processing
#from crseg_metrics import varifold_distance,mesh,mesh_explicit
from varifold import mesh,mesh_explicit
from skimage import morphology, filters
from scipy.ndimage import gaussian_filter
from crseg_utils import resample,mean_threshold,crop_around_COM,\
    align_COM_masks,res_match_and_rescale,max_threshold,normalize_volume,\
    unpad,resize,uncrop_volume,join_labels
from torchmcubes import marching_cubes
from utils import print_no_newline


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../airlab/airlab/")))) #add airlab dir to working dir
import airlab as al


"""
Main CRSEG registration function. This framework and basis-splines
transformation model is directly based on AIRLAB functions:

[2018] Robin Sandkuehler, Christoph Jud, Simon Andermatt, and Philippe C.
Cattin. "AirLab: Autograd Image Registration Laboratory". arXiv preprint
arXiv:1806.09907, 2018.

"""

def register_multichan(casepath,
                     fixed_image_list,
                     moving_image_list,
                     fixed_mask_list,
                     moving_mask_list,
                     fixed_loss_region,
                     moving_loss_region,
                     regularisation_weight=[1.0,2.0,3.0,4.0],
                     mask_weights=None,
                     n_iters=[500, 400, 200, 180],
                     step_size=[4e-2, 2e-3, 5e-4, 8e-5],
                     pyramid_ds=[8,4,2],
                     pyramid_sigma=[6,6,6,8],
                     mesh_spacing=[3, 3, 3, 3],
                     relax_alpha=3.0,
                     relax_beta=1.5,
                     affine_step=False,
                     diffeomorphic=True,
                     using_masks=True,
                     use_single_channel=False,
                     relax_regularization=True,
                     no_superstructs=False,
                     use_varifold=True,
                     varifold_sigma=30.0,
                     use_MSE=False,
                     use_Dice=False,
                     spline_order=3):


    print("===========================================================")
    print("STARTING CRSEG REGISTRATION")
    print("    - Regularization weight: ", regularisation_weight)
    print("    - Downsampling factor: ", pyramid_ds)
    print("    - Number of iterations: ", n_iters)
    print("    - Step size: ", step_size)
    print("    - Diffeomorphic: ", diffeomorphic)
    print("    - Relaxed regularization: ", relax_regularization)
    print("    - Pyramid sigma vals: ", pyramid_sigma)
    print("    - Spline order: ", spline_order)
    print("===========================================================")


    dtype = th.float32
    device = th.device("cpu")

    print("Building image pyramid...")

    fixed_image_levels_list = []
    moving_image_levels_list = []
    fixed_mask_levels_list = []
    moving_mask_levels_list = []

    """
        creating a separable pyramid for each volume set and mask set
    """
    for i in range(0,len(pyramid_ds)+1):
        fixed_mask_pyramid_list = []
        moving_mask_pyramid_list = []
        for j in range(0,len(fixed_mask_list)):
            fixed_mask_pyramid_list.append(al.create_image_pyramid(fixed_mask_list[j], [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])[i])
            moving_mask_pyramid_list.append(al.create_image_pyramid(moving_mask_list[j], [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])[i])
        fixed_mask_levels_list.append(fixed_mask_pyramid_list)
        moving_mask_levels_list.append(moving_mask_pyramid_list)

    # create channel-split image pyramid size/8 size/4, size/2, size/1
    for i in range(0,len(pyramid_ds)+1):
        fixed_image_pyramid_list = []
        moving_image_pyramid_list = []
        for j in range(0,len(fixed_image_list)):
            fixed_image_pyramid_list.append(al.create_image_pyramid(fixed_image_list[j], [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])[i])
            moving_image_pyramid_list.append(al.create_image_pyramid(moving_image_list[j], [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])[i])
        fixed_image_levels_list.append(fixed_image_pyramid_list)
        moving_image_levels_list.append(moving_image_pyramid_list)


    print_no_newline("finished building pyramid... ")
    fixed_loss_region_pyramid = al.create_image_pyramid(fixed_loss_region, [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])
    moving_loss_region_pyramid = al.create_image_pyramid(moving_loss_region, [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])
    sigma = [[pyramid_sigma[0],pyramid_sigma[0],pyramid_sigma[0]], [pyramid_sigma[1],pyramid_sigma[1],pyramid_sigma[1]], [pyramid_sigma[2],pyramid_sigma[2],pyramid_sigma[2]], [pyramid_sigma[3],pyramid_sigma[3],pyramid_sigma[3]]]
    constant_flow = None
    print("done")


    for level, (fix_loss_reg_level, mov_loss_reg_level) in enumerate(zip(fixed_loss_region_pyramid, moving_loss_region_pyramid)):
        print("---- Level "+str(level)+" ----")
        print("adjusting weighting to: ", mask_weights[level])
        registration = al.PairwiseRegistration(verbose=True)


        # define the spline transformation object
        transformation = al.transformation.pairwise.BsplineTransformation(moving_image_levels_list[level][0].size,
                                                                          diffeomorphic=diffeomorphic,
                                                                          sigma=sigma[level],
                                                                          order=spline_order,
                                                                          dtype=dtype,
                                                                          device=device)

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                          moving_image_levels_list[level][0].size,
                                                                          interpolation="linear")

            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)



        fix_im_list_level = fixed_image_levels_list[level]
        mov_im_list_level = moving_image_levels_list[level]
        fix_msk_list_level = fixed_mask_levels_list[level]
        mov_msk_list_level = moving_mask_levels_list[level]


        if use_varifold == True:
            ## create varifold mesh for this pyramid level
            print_no_newline("meshing... ")

            #moving_verts_list = []
            #moving_faces_list = []
            #fixed_cts_list = []
            #fixed_norms_list = []
            #mesh_weight = 0
            #for i in range(len(fix_msk_list_level)):
            #    moving_verts,moving_faces,_,_ = mesh(mean_threshold(mov_msk_list_level[i].numpy()),step_size=mesh_spacing[level])
            #    moving_verts_list.append(moving_verts)
            #    moving_faces_list.append(moving_faces)

            #    _,_,fixed_cts,fixed_norms = mesh(mean_threshold(fix_msk_list_level[i].numpy()),step_size=mesh_spacing[level])
            #    fixed_cts_list.append(fixed_cts)
            #    fixed_norms_list.append(fixed_norms)
            #    mesh_weight += np.max(fixed_norms.shape)
            fixed_norms_list=None
            fixed_cts_list=None
            moving_verts_list=None
            moving_faces_list=None
            print("done")
            print("Adjusted mesh spacing to: ", mesh_spacing[level])

            #mask_weights[level] /= mesh_weight
        else:
            fixed_norms_list=None
            fixed_cts_list=None
            moving_verts_list=None
            moving_faces_list=None

        image_loss = al.loss.pairwise.COMPOUND(casepath,
                                            fix_im_list_level,
                                            mov_im_list_level,
                                            fix_loss_reg_level,
                                            fix_msk_list_level,
                                            mov_loss_reg_level,
                                            mov_msk_list_level,
                                            loss_region_weight=1.0,
                                            channel_weight=1.0,
                                            mask_weights=mask_weights[level],
                                            varifold=use_varifold,
                                            vf_sigma=varifold_sigma,
                                            step_size=mesh_spacing[level],
                                            MSE=use_MSE,
                                            Dice=use_Dice,
                                            epsilon=None,
                                            single_channel=use_single_channel,
                                            no_superstructs=no_superstructs,
                                            generate_mesh=False,
                                            fixed_norms_list=fixed_norms_list,
                                            fixed_cts_list=fixed_cts_list,
                                            moving_verts_list=moving_verts_list,
                                            moving_faces_list=moving_faces_list)


        registration.set_image_loss([image_loss])

        # define the regulariser for the displacement
        print("using relaxed diffusion regularizer...setting regularizer weighting to: ",regularisation_weight[level])
        regulariser = al.regulariser.displacement.Relaxed_DiffusionRegulariser(moving_image_levels_list[level][0].spacing,
                                                                            mov_msk_list_level,
                                                                            alpha=relax_alpha,
                                                                            beta=relax_beta,
                                                                            using_masks=using_masks,
                                                                            relax_regularization=relax_regularization)
        regulariser.SetWeight(regularisation_weight[level])

        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = th.optim.Adam(transformation.parameters(), lr=step_size[level], amsgrad=True)
        #optimizer = th.optim.LBFGS(transformation.parameters(), lr=step_size[level])

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(n_iters[level])

        registration.start()

        # store current flow field
        constant_flow = transformation.get_flow()
        current_displacement = transformation.get_displacement()
        # generate SimpleITK displacement field and calculate TRE
        tmp_displacement = al.transformation.utils.upsample_displacement(current_displacement.clone().to(device='cpu'),
                                                                    moving_image_levels_list[level][0].size, interpolation="linear")
        tmp_displacement = al.transformation.utils.unit_displacement_to_displacement(tmp_displacement)  # unit measures to image domain measures
        tmp_displacement = al.create_displacement_image_from_image(tmp_displacement,  moving_image_levels_list[level][0])
        tmp_displacement.write('/tmp/bspline_displacement_image_level_'+str(level)+'.vtk')
        print("===================================================================================")
        print("                        LEVEL: ", level, " DONE!")
        print("===================================================================================")

    # create final result
    displacement = transformation.get_displacement()


    warped_test_image = al.transformation.utils.warp_image(moving_image_levels_list[level][0], displacement)
    displacement_out = displacement.clone() # remember to copy or deepcopy tensors
    #displacement_out = al.transformation.utils.unit_displacement_to_displacement(displacement_out)
    #np.save(os.path.join(casepath, "scratch/displacement.npy"), displacement_out)
    displacement = al.transformation.utils.unit_displacement_to_displacement(displacement) # unit measures to image domain measures
    displacement = al.create_displacement_image_from_image(displacement, moving_image_levels_list[level][0])

    print("=================================================================")
    print("               CRSEG registration finished!")
    print("=================================================================")

    return displacement_out, warped_test_image





"""

Main function for propagating labels along the calculated deformation
field and saving the binarized propagations..

"""



def propagate(label_path,
          test_wm_seg_path,
          atlas_wm_mask_path,
          savepath,
          scratch_dir,
          atlas_header,
          displacement,
          save_affines=[None,None],
          resample_factor=None,
          flip=False,
          overlap=0.5,
          resolution_flip=False,
          rotate_atlas=True,
          r_atlas=0.5,
          r_test=1.0,
          r_test_base=1.0,
          speed_crop=True,
          tolerance=[50,50],
          pre_affine_step=True):



    print("Propagating labels...")

    # volume extensions to consider
    extensions = ('.nii','.nii.gz','.mgz')

    [save_affine_test,save_affine_atlas] = save_affines
    ## collect AAN label file names for later concatenation
    label_name_collector = []

    #### set glob params and import stuff ####
    dtype = th.float32
    device = th.device("cpu")
    atlas_mask_foo = nib.load(atlas_wm_mask_path)
    atlas_mask = np.array(atlas_mask_foo.dataobj)
    # strictly for finding COM
    #joint_wm_atlas_mask = join_labels(np.rot90(atlas_mask,k=3,axes=(1, 2)))
    joint_wm_atlas_mask = join_labels(atlas_mask)

    for subdir, dirs, files in os.walk(label_path):
        for file in files:
            if len(file.split('.')) > 2:
                ext = '.' + file.split('.')[1] + '.' + file.split('.')[2]
            else:
                ext = '.' + file.split('.')[1]

            if ext in extensions:
                print("found: ",os.path.join(subdir, file))

                ###### work the input labl #######
                vol_foo = nib.load(os.path.join(subdir, file))
                vol = np.array(vol_foo.dataobj)

                if rotate_atlas is True:
                    vol = np.rot90(vol,k=3,axes=(1, 2))


                if not pre_affine_step:
                    vol_rescaled,_, test_p_matrix = res_match_and_rescale(vol,
                                                                    test_mask,
                                                                    r_atlas,
                                                                    r_test_base,
                                                                    resample_factor,
                                                                    res_flip=resolution_flip)

                    vol_rescaled = max_threshold(vol_rescaled)

                    vol_aligned,_ = align_COM_masks(vol_rescaled,
                                                    vol_rescaled,
                                                    test_mask_rescaled,
                                                    atlas_mask_rescaled)
                else:
                    vol_aligned = vol

                if speed_crop:
                    _,vol_cropped,_,_,c_mat1,c_mat2 = crop_around_COM(vol_aligned,
                                                                  vol_aligned,
                                                                  joint_wm_atlas_mask,
                                                                  joint_wm_atlas_mask,
                                                                  tolerance)
                else:
                    vol_cropped = vol_aligned


                nib.save(nib.Nifti1Image(vol_cropped,affine=save_affine_test), os.path.join(savepath,file.replace(ext,'_cropped.nii.gz')))
                #####################################################################################

                [s0, s1, s2] = vol_cropped.shape
                air_vol = al.Image(vol_cropped.astype(np.float32), [s0, s1, s2], [1, 1, 1], [0, 0, 0])

                ### propagate the original volume
                warped_air_vol = al.transformation.utils.warp_image(air_vol, displacement)

                if speed_crop:
                    vol_uncropped = uncrop_volume(warped_air_vol.numpy(),c_mat1)
                else:
                    vol_uncropped = warped_air_vol.numpy()
                if resolution_flip and not pre_affine_step:
                    warped_vol_unpadded = unpad(vol_uncropped, test_p_matrix)
                    warped_air_vol_ds = resize(warped_vol_unpadded,(pad_shape[0],pad_shape[1],pad_shape[2]),anti_aliasing=True)
                else:
                    warped_air_vol_ds = vol_uncropped

                val = filters.threshold_otsu(warped_air_vol_ds)
                vol_np = warped_air_vol_ds
                vol_np[vol_np <= val*overlap] = 0
                vol_np = vol_np/np.amax(vol_np,axis=(0,1,2))
                vol_np[vol_np <= overlap] = 0
                vol_np[vol_np > 0]  = 1

                #if not resolution_flip:
                #    vol_np = unpad(vol_np, test_p_matrix)

                #if "_L" not in file:
                #    if "_R" not in file:
                #        print("adding to label volume")
                #        empty_vol[vol_np == 1] = counter
                #        counter += 1

                #vol_nib = nib.Nifti1Image(vol_np,affine=atlas_mask_foo.affine)
                nib.save(nib.Nifti1Image(vol_np,affine=save_affine_atlas), os.path.join(savepath,file.replace(ext,'_warped_MNI_space.nii.gz')))

                os.system("reg_resample -inter 0 -flo " + os.path.join(savepath,file.replace(ext,'_warped_MNI_space.nii.gz')) + " -ref " + os.path.join(scratch_dir,"b0_test.nii.gz") + " -trans " + os.path.join(scratch_dir,"inverse_affine_mat.txt") + " -res " + os.path.join(savepath,file.replace(ext,'_inverse_transformed.nii.gz')))
                label_name_collector.append(os.path.join(savepath,file.replace(ext,'_inverse_transformed.nii.gz')))


    print_no_newline("creating and saving joint label volume... ")
    ### collect all transformed labels into a single array and match to dictionary accordingly
    aan_label_dict = {1001: 'DR',1002: 'PAG',1003: 'MnR',1004: 'VTA',1005: 'LC_L',2005: 'LC_R',1006: 'LDTg_L',2006: 'LDTg_R',1007: 'PBC_L',2007: 'PBC_R',1008: 'PnO_L',2008: 'PnO_R',1009: 'mRt_L',2009: 'mRt_R',1011: 'PTg_L',2011: 'PTg_R'}
    #aan_label_dict = {1001: 'label_1001',1002: 'label_1002',1003: 'label_1003',1004: 'label_1004',1005: 'label_1005', 1006: 'label_1006', 1007: "label_1007", 2001: 'label_2001',2002: 'label_2002',2003: 'label_2003',2004: 'label_2004',2005: 'label_2005',2006: 'label_2006'}


    foovol = nib.load(label_name_collector[0])
    all_label_vol = np.zeros_like(np.array(foovol.dataobj))

    for lab in label_name_collector:
        for key, value in aan_label_dict.items():
            if re.search(value, lab, re.IGNORECASE):
                label_vol = nib.load(lab)
                label_vol_np = np.array(label_vol.dataobj)
                all_label_vol[label_vol_np > 0] = key


    all_label_vol_nib = nib.Nifti1Image(all_label_vol,affine=foovol.affine)
    nib.save(all_label_vol_nib,os.path.join(savepath,"AAN_label_volume_transformed.nii.gz"))
    print("done")
