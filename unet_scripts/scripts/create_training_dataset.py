import os,sys,shutil
import numpy as np
import random


########
# CURRENTLY ONLY APPLIES FOR HCP!!!
########

data_dir = "/autofs/vast/nicc_003/users/olchanyi/data/HCP100"
output_dir = "./training_dataset"
training_dir = os.path.join(output_dir,"train")
validation_dir = os.path.join(output_dir,"validate")

## between 0 (all validate) and 1 (all train)
training_ratio = 0.75

if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
os.makedirs(training_dir)
os.makedirs(validation_dir)

data_list = next(os.walk(data_dir))[1]
random.shuffle(data_list)

for subject_name in data_list:
        subject_dir = os.path.join(data_dir,subject_name)
        if os.path.exists(os.path.join(subject_dir,"training_labels/ROI00.nii.gz")):
                if random.uniform(0, 1) > training_ratio:
                        output_subject_dir = os.path.join(validation_dir,"subject_" + subject_name)
                else:
                        output_subject_dir = os.path.join(training_dir,"subject_" + subject_name)
                os.makedirs(output_subject_dir)
                os.makedirs(os.path.join(output_subject_dir,"dmri"))
                os.makedirs(os.path.join(output_subject_dir,"segs"))

                ## make np label list
                label_array = np.asarray([1])
                np.save(os.path.join(output_subject_dir,"label_list.npy"),label_array)

                aseg_file = os.path.join(subject_dir,"training_labels/ROI00.nii.gz")
                t1_file = os.path.join(subject_dir, 'T1w/Diffusion', 'lowb.nii')
                fa_file = os.path.join(subject_dir, 'T1w/Diffusion/dmri', 'dtifit.1K_FA.nii.gz')
                v1_file = os.path.join(subject_dir, 'mrtrix_outputs', 'tracts_concatenated.nii.gz')

                print("copying volumes for: ", subject_name)
                ## move to respective training folders
                os.system("cp " + t1_file + " " + os.path.join(output_subject_dir,'subject_' + subject_name + '.t1.nii'))
                os.system("cp " + fa_file + " " + os.path.join(output_subject_dir,"dmri",'subject_' + subject_name + '_1k_fa.nii.gz'))
                os.system("cp " + v1_file + " " + os.path.join(output_subject_dir,"dmri",'subject_' + subject_name + '_1k_v1.nii.gz'))
                os.system("cp " + aseg_file + " " + os.path.join(output_subject_dir,"segs",'subject_' + subject_name + '_1k_DSWbeta.nii.gz'))
