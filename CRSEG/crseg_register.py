import sys
import shutil
import os
import math
import numpy as np
import torch as th
from dipy.io.image import load_nifti, save_nifti
#from crseg_metrics import varifold_distance,mesh,mesh_explicit
from varifold import mesh,mesh_explicit
from skimage import morphology, filters
from scipy.ndimage import gaussian_filter
from crseg_utils import resample,mean_threshold,crop_around_COM,\
    align_COM_masks,res_match_and_rescale,max_threshold,normalize_volume,\
    unpad,resize,uncrop_volume
from torchmcubes import marching_cubes


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



    fixed_loss_region_pyramid = al.create_image_pyramid(fixed_loss_region, [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])
    moving_loss_region_pyramid = al.create_image_pyramid(moving_loss_region, [[pyramid_ds[0],pyramid_ds[0],pyramid_ds[0]], [pyramid_ds[1],pyramid_ds[1],pyramid_ds[1]], [pyramid_ds[2],pyramid_ds[2],pyramid_ds[2]]])
    sigma = [[pyramid_sigma[0],pyramid_sigma[0],pyramid_sigma[0]], [pyramid_sigma[1],pyramid_sigma[1],pyramid_sigma[1]], [pyramid_sigma[2],pyramid_sigma[2],pyramid_sigma[2]], [pyramid_sigma[3],pyramid_sigma[3],pyramid_sigma[3]]]
    constant_flow = None
    print("finished building pyramid...")


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
            print("meshing... adjusting mesh spacing to: ", mesh_spacing[level])


            moving_verts_list = []
            moving_faces_list = []
            fixed_cts_list = []
            fixed_norms_list = []
            mesh_weight = 0
            for i in range(len(fix_msk_list_level)):
                moving_verts,moving_faces,_,_ = mesh(mean_threshold(mov_msk_list_level[i].numpy()),step_size=mesh_spacing[level])
                moving_verts_list.append(moving_verts)
                moving_faces_list.append(moving_faces)

                _,_,fixed_cts,fixed_norms = mesh(mean_threshold(fix_msk_list_level[i].numpy()),step_size=mesh_spacing[level])
                fixed_cts_list.append(fixed_cts)
                fixed_norms_list.append(fixed_norms)
                mesh_weight += np.max(fixed_norms.shape)
                print("number of fixed total mesh elements for moving mask ",i, " : ", np.max(moving_faces.shape))
                print("number of fixed total mesh elements for fixed mask ",i, " : ", np.max(fixed_norms.shape))

            mask_weights[level] /= mesh_weight
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



def propagate(path,
          savepath,
          basepath,
          savepath_normalized,
          displacement,
          test_p_matrix_foo,
          test_mask_path,
          atlas_mask_path,
          dummy_affine,
          resample_factor=None,
          flip=False,
          overlap=0.5,
          resolution_flip=True,
          resample_input=False,
          rotate_atlas=True,
          r_atlas=0.5,
          r_test=0.75,
          r_test_base=0.75,
          speed_crop=True,
          tolerance=[50,50],
          pre_affine_step=True):


    print("=================================================================")
    print("               Beginning label propagation...")
    print("=================================================================")

    # extensions to consider
    rootdir = path
    extensions = ('.nii','.nii.gz','.mgz')


    if resample_input is not True:
        res_test_base = r_test
    else:
        res_test_base = r_test_base


    #### set glob params and import stuff ####
    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")


    counter = 1
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if len(file.split('.')) > 2:
                ext = '.' + file.split('.')[1] + '.' + file.split('.')[2]
            else:
                ext = '.' + file.split('.')[1]

            if ext in extensions:
                print("found: ",os.path.join(subdir, file))

                #####################################################################################

                test_mask, affine_tm = load_nifti(test_mask_path, return_img=False)
                test_volume_foo = test_mask.copy()
                pad_shape = test_mask.shape


                if resample_input is True:
                    print("resampling to: ", res_test, " mm")
                    test_mask = resample(test_mask,res_atlas,res_test)
                    test_mask = mean_threshold(test_mask_resampled)

                if flip is True:
                    print("flipping target volumes")
                    test_mask = np.flip(test_mask,1)

                atlas_mask,_ = load_nifti(atlas_mask_path, return_img=False)

                if rotate_atlas is True:
                    atlas_mask = np.rot90(atlas_mask,k=3,axes=(1, 2))
                    atlas_mask = max_threshold(atlas_mask)
                else:
                    atlas_mask = max_threshold(atlas_mask)



                if not pre_affine_step:
                    atlas_mask_rescaled, test_mask_rescaled, mask_p_matrix = res_match_and_rescale(atlas_mask,
                                                                                        test_mask,
                                                                                        r_atlas,
                                                                                        r_test_base,
                                                                                        resample_factor,
                                                                                        res_flip=resolution_flip)





                    test_mask_rescaled= max_threshold(test_mask_rescaled)
                    atlas_mask_rescaled = max_threshold(atlas_mask_rescaled)

                    aligned_atlas_mask,_ = align_COM_masks(test_mask_rescaled,
                                                        atlas_mask_rescaled,
                                                        test_mask_rescaled,
                                                        atlas_mask_rescaled)


                else:
                    test_mask_rescaled = test_mask
                    aligned_atlas_mask = atlas_mask


                ###### work the input labl #######

                vol, affine_v = load_nifti(os.path.join(subdir, file), return_img=False)


                if rotate_atlas is True:
                    vol = np.rot90(vol,k=3,axes=(1, 2))
                    vol = max_threshold(vol)
                else:
                    vol = max_threshold(vol)


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
                    _,vol_cropped,_,_,c_mat1,c_mat2 = crop_around_COM(test_mask_rescaled,
                                                                  vol_aligned,
                                                                  test_mask_rescaled,
                                                                  aligned_atlas_mask,
                                                                  tolerance)



                #####################################################################################

                [s0, s1, s2] = vol_cropped.shape
                air_vol = al.Image(vol_cropped, [s0, s1, s2], [1, 1, 1], [0, 0, 0])


                # translate moving im to center of mass of PM mask
                if counter == 1:
                    empty_vol = np.zeros_like(test_volume_foo)

                ### propagate the original volume
                warped_air_vol = al.transformation.utils.warp_image(air_vol, displacement)

                if speed_crop:
                    vol_uncropped = uncrop_volume(warped_air_vol.numpy(),c_mat1)

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

                if not resolution_flip:
                    vol_np = unpad(vol_np, test_p_matrix)

                if "_L" not in file:
                    if "_R" not in file:
                        print("adding to label volume")
                        empty_vol[vol_np == 1] = counter
                        counter += 1

                save_nifti(savepath + file,vol_np,dummy_affine)
                os.system("reg_resample -inter 0 -flo " + savepath + file + " -ref " + basepath + "/b0.nii.gz -trans " + basepath + "/scratch/inverse_affine_mat.txt -res " + savepath_normalized + file)

                ######### preliminary dice scores if gold-standard labels exists:
                gs_file = basepath + "/gold_standard_labels/" + file
                if os.path.exists(gs_file):
                    print("GOLD STANDARD LABEL EXISTS: SEE BELOW")
                    os.system("mri_seg_overlap " + gs_file + " " + savepath_normalized + file)
                    print("===========================================================================")
    save_nifti(savepath + "label_volume.nii.gz",empty_vol,dummy_affine)
