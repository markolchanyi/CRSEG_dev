import os, sys, argparse, datetime, torch, subprocess
from subprocess import Popen,PIPE
import math, numpy as np, nibabel as nb, xlsxwriter, pandas as pd
np.set_printoptions(threshold=sys.maxsize) #Allows full matrix to be visable instead of truncated
path = os.getcwd()


def parse_args_probtrackx():
    parser = argparse.ArgumentParser(description="ptrackx stuff.")
    parser.add_argument('-b','--bedpost_path', help="bedpostx directory", type=str, required=True)
    parser.add_argument('-s','--seg_path', help="AAN segmentation path", type=str, required=True)
    parser.add_argument('-p','--probtrackx_path', help="output probtrackx path", type=str, required=True)
    parser.add_argument('-t','--template_path', help="template path", type=str, required=True)


def get_roi_list(roi_path):
    rois = []
    print("creating ROI network list for Probtrackx...")
    for item in [os.path.join(roi_path, file) for file in sorted(os.listdir(roi_path))]:
        if item.endswith(".nii"):
            print("appending " + item)
            rois.append(item)
    network_file_name = "probtrackx_roi_network.txt"
    network_file = os.path.join(roi_path, network_file_name)

    with open(network_file, 'w') as f:
        for item in rois:
            f.write("%s\n" % item)
    return network_file


def set_device():
    if torch.cuda.is_available():
        #os.environ["CUDA_HOME"]="/usr/pubsw/packages/CUDA/11.6/"
        os.environ["CUDA_HOME"]="/usr/pubsw/packages/CUDA/9.1/"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        #os.environ["LD_LIBRARY_PATH"]="/usr/pubsw/packages/CUDA/11.6/lib64:/usr/pubsw/packages/CUDA/11.6/extras/CUPTI/lib64:/usr/pubsw/packages/CUDA/9.0/lib64:/usr/pubsw/packages/CUDA/9.1/lib64"
        device = torch.device("cuda")
        cuda = "1"
    else:
        device = torch.device("cpu")
    return(device, cuda)


def set_device_new():
    if torch.cuda.is_available():
        #os.environ["CUDA_HOME"]="/usr/pubsw/packages/CUDA/11.6/"
        os.environ["CUDA_HOME"]="/usr/pubsw/packages/CUDA/10.2/"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        os.environ["LD_LIBRARY_PATH"]="/usr/pubsw/packages/CUDA/10.2/lib64"
        device = torch.device("cuda")
        cuda = "1"
    else:
        device = torch.device("cpu")
    return(device, cuda)



def set_geometry(bedpost_directory, network_file):
    """
    Copies diffusion mask geometry header and applies it to all ROIs in list. This is to prevent possible mismatched matrices that is often caused by the geometry headers not matching. This does not change or alter the data besides updating the header.
    """

    mask = os.path.join(bedpost_directory, "nodif_brain_mask.nii.gz")

    if os.path.exists(mask) == True and os.path.exists(network_file) == True:
        print("Diffusion Mask Volume == ", mask)
        print("ROI Network List == ", network_file)

        roi = open(network_file, 'r')
        rois = roi.readlines()

        for line in rois:
            print("Working on", line)
            geometryCommand = "fslcpgeom " + mask + " " + line
            os.system(geometryCommand)


def voxel_sizes(roi_input):
    """
    Returns the non-zero voxel size
    (total count and volume mm^3) for the given input roi. By default only returns non-zero voxel count
    """

    ##LOAD DATA
    print(":::: VOXEL SIZES ::::")
    nii = nb.load(roi_input)
    img = nii.get_fdata()

    ##GET DIMENSIONS
    voxel_dims = (nii.header["pixdim"])[1:4]

    ##COMPUTE VOLUME
    nonzero_voxel_count = np.count_nonzero(img)
    voxel_volume = np.prod(voxel_dims)
    nonzero_voxel_volume = nonzero_voxel_count * voxel_volume

    print("NON ZERO VOXELS = {}".format(nonzero_voxel_count))
    print("VOLUME NON ZERO VOXELS = {} mm^3".format(nonzero_voxel_volume))

    return nonzero_voxel_count

def write_excel_head(headers_list,row,column):
    """
    Writes the header row into the spreadsheet.
    """
    ##WRITE HEADER ROW
    for header in headers_list:
        worksheet.write(row, column, header)
        column += 1

def write_excel_oneway(roi_row, roi_column, label, vox, streamlines, weighted_vox):
    """
    Function to write individual ROI statistics into designated space in the spreadsheet.
    """
    ##POPULATE LIST TO LOOK THROUGH WRITER
    #items_to_write=(label, vox, streamlines, weighted_vox)

    #for x in items_to_write:
    #    worksheet.write(roi_row, roi_column, x)
    #    roi_column += 1
    items_to_write=(label, vox, streamlines, weighted_vox)

    for x in items_to_write:
        if isinstance(x, str):
            worksheet.write(roi_row, roi_column, x)
        else:
            if math.isnan(x) or math.isinf(x):
                worksheet.write(roi_row, roi_column, 'NaN or INF')
            else:
                worksheet.write(roi_row, roi_column, x)
        roi_column += 1

def oneway_calc(matrix, index, roi_list, roi, voxels, row, column, header_list):
    """
    Main function to calculate all connectivity statistics for each relationship.
    Contains calls to additional functions to perform targeted calculations.
    """

    single_seed_array = matrix[index]
    single_seed_array_list = single_seed_array.tolist()
    TARGET2SEED_ARRAY = [column[index] for column in matrix]

    for target in roi_list:

        ##SEED TO TARGET CALCULATIONS
        print(":::: SEED-2-TARGET RELATIONSHIP ::::")
        print("CURRENT SEED ROW::: ", row)
        direction = "CPa"
        target_basename = os.path.basename(target)
        seed_basename = os.path.basename(roi)
        label = (seed_basename + ".2." + target_basename)
        target_position = roi_list.index(target)

        RAW_SEED_VOXELS = (voxels)
        SEED2TARGET_STREAMLINES = single_seed_array[target_position]
        WEIGHTED_SEED_VOXELS = (RAW_SEED_VOXELS * 5000)
        print("RELATIONSHIP =", label)
        print("SEED STREAMLINES=", SEED2TARGET_STREAMLINES)
        print("SEED VOXELS =", WEIGHTED_SEED_VOXELS)
        print()

        write_excel_oneway(row, column, label, RAW_SEED_VOXELS, SEED2TARGET_STREAMLINES, WEIGHTED_SEED_VOXELS)
        oneway_connectivity_probability(header_list, row, SEED2TARGET_STREAMLINES, WEIGHTED_SEED_VOXELS, direction)

        ##TARGET TO SEED CALCULATIONS
        print(":::: TARGET-2-SEED RELATIONSHIP ::::")
        column = 4
        direction = "CPb"
        TARGET2SEED_STREAMLINES = TARGET2SEED_ARRAY[target_position]
        inverse_label = (target_basename + ".2." + seed_basename)
        targ_nonzero_vox_count=voxel_sizes(target)
        weighted_seed_voxels=(targ_nonzero_vox_count * 5000)
        print("RELATIONSHIP =", inverse_label)
        print("WEIGHTED SEED VOXELS =", weighted_seed_voxels)
        print("TARGET STREAMLINES=", TARGET2SEED_STREAMLINES)
        print()

        write_excel_oneway(row, column, inverse_label, targ_nonzero_vox_count, TARGET2SEED_STREAMLINES, weighted_seed_voxels)
        oneway_connectivity_probability(header_list, row, TARGET2SEED_STREAMLINES, weighted_seed_voxels, direction)
        round_connectivity_probability(HEADERS, row, SEED2TARGET_STREAMLINES, TARGET2SEED_STREAMLINES, WEIGHTED_SEED_VOXELS, weighted_seed_voxels)

        column = 0
        row += 1
        print()

def oneway_connectivity_probability(headers_list,row,streamlines,voxel_weight,direction):
    """
    Calculates the one-way statistical probability of connectivity between identified seed and target.
    """
    #if direction == 'CPa':
    #    column = headers_list.index('CPa')
    #    SEED2TARG_CP = (streamlines / voxel_weight)
    #    worksheet.write(row,column, SEED2TARG_CP)

    #if direction == 'CPb':
    #    column = headers_list.index('CPb')
    #    TARG2SEED_CP = (streamlines / voxel_weight)
    #    worksheet.write(row,column, TARG2SEED_CP)

    if direction == 'CPa':
        column = headers_list.index('CPa')
        if isinstance(streamlines, str) or isinstance(voxel_weight, str):
            worksheet.write(row, column, 'NaN or INF')
        elif math.isnan(streamlines) or math.isnan(voxel_weight) or voxel_weight == 0:
            worksheet.write(row, column, 'NaN or INF')
        else:
            SEED2TARG_CP = streamlines / voxel_weight
            worksheet.write(row, column, SEED2TARG_CP)

    if direction == 'CPb':
        column = headers_list.index('CPb')
        if isinstance(streamlines, str) or isinstance(voxel_weight, str):
            worksheet.write(row, column, 'NaN or INF')
        elif math.isnan(streamlines) or math.isnan(voxel_weight) or voxel_weight == 0:
            worksheet.write(row, column, 'NaN or INF')
        else:
            TARG2SEED_CP = streamlines / voxel_weight
            worksheet.write(row, column, TARG2SEED_CP)


def round_connectivity_probability(headers_list, row, seed_streamline, target_streamline, seed_vox_value, targ_vox_value):
    """
    Calculates the overall, weighted statistical probability of connectivity between seed to target, and target to seed.
    """
    #column = headers_list.index('CPab')
    #TOTAL_STREAMLINE = (seed_streamline + target_streamline)
    #TOTAL_WEIGHTED_VOX = (seed_vox_value + targ_vox_value)
    #TOTAL_CP = (TOTAL_STREAMLINE / TOTAL_WEIGHTED_VOX)
    #print(TOTAL_CP)

    #worksheet.write(row,column,TOTAL_CP)

    column = headers_list.index('CPab')
    if (isinstance(seed_streamline, str) or isinstance(target_streamline, str) or
            isinstance(seed_vox_value, str) or isinstance(targ_vox_value, str) or
            math.isnan(seed_streamline) or math.isnan(target_streamline) or
            math.isnan(seed_vox_value) or math.isnan(targ_vox_value) or
            seed_vox_value == 0 or targ_vox_value == 0):
        worksheet.write(row, column, 'NaN or INF')
    else:
        TOTAL_STREAMLINE = seed_streamline + target_streamline
        TOTAL_WEIGHTED_VOX = seed_vox_value + targ_vox_value
        TOTAL_CP = TOTAL_STREAMLINE / TOTAL_WEIGHTED_VOX
        worksheet.write(row, column, TOTAL_CP)











args = parse_args_probtrackx()

bedpost_path = args.bedpost_path
seg_path = args.seg_path
probtrackx_base_path = args.probtrackx_path
template_path = args.template_path

roi_path = os.path.join(probtrackx_base_path,"rois")
if not os.path.exists(roi_path):
    os.makedirs(roi_path)

aan_label_dict = {1001: 'DR',1002: 'PAG',1003: 'MnR',1004: 'VTA',1005: 'LC_L',2005: 'LC_R',1006: 'LDTg_L',2006: 'LDTg_R',1007: 'PBC_L',2007: 'PBC_R',1008: 'PnO_L',2008: 'PnO_R',1009: 'mRt_L',2009: 'mRt_R',1011: 'PTg_L',2011: 'PTg_R'}

for key, value in aan_label_dict.items():
    print("looking for " + key + " with value " + str(value) + ". saving as " + os.path.join(roi_path,key + ".nii.gz"))
    os.system("mri_binarize --noverbose --i " + seg_path + " --o " + os.path.join(roi_path,key + ".nii.gz") + " --match " + str(key))
    os.system("mri_convert " + os.path.join(roi_path,key + ".nii.gz") + " " + os.path.join(roi_path,key + ".nii") + " -rl " + template_path + " -rt nearest -odt float")
    os.system("rm " + os.path.join(roi_path,key + ".nii.gz"))





### ------------- RUN PROBTRACKX -------------- ###
device,cuda = set_device()
network_file = get_roi_list(roi_path)
set_geometry(bedpost_path,network_file)
print("------------------- Setting Up Enviornment -------------------")
os.environ["FSLDIR"] = "/usr/pubsw/packages/fsl/6.0.5.1"

print("FSL DIRECTORY --> ", os.environ.get("FSLDIR"))
print("CUDA HOME --> ", os.environ.get("CUDA_HOME"))
print("LD LIBRARY PATH --> ", os.environ.get("LD_LIBRARY_PATH"))
print("CUDA VISIBLE DEVICES --> ", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(" ")

probtrackx_path = os.path.join(probtrackx_base_path,"outputs")
if not os.path.exists(probtrackx_path):
    os.makedirs(probtrackx_path)

probtrackxCommand= "/usr/pubsw/packages/fsl/6.0.5.1/bin/probtrackx2_gpu" + " -x " + network_file + " -s " + os.path.join(bedpost_path, "merged") + " -m " + os.path.join(bedpost_path, "nodif_brain_mask") + " --dir=" + probtrackx_path + " -l -c 0.2 -S 2000 -P 5000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --steplength=0.25 --forcedir --opd --network -V 1"
os.system(probtrackxCommand)
print("done with everything!")







#-----------------------------------------------------------------------------------------
##VARIABLES
ROI_FILE = network_file
# For example see: '/autofs/space/nicc_002/EXC/EXC007/Bay3/scripts/dmn_network_run.txt'
#NETWORK_MATRIX = '/autofs/space/nicc_003/users/olchanyi/shared/for_holly/EXC_RO1_annotations_20230625/EXC030/exc030.probtrackx.network.20230626/fdt_network_matrix'
NETWORK_MATRIX = os.path.join(probtrackx_path,"fdt_network_matrix")
# For example see: '/autofs/space/nicc_002/EXC/EXC007/Bay3/diff.probtrackx2.network/fdt_network_matrix'
SUBJECT = 'CP'
# Will be used to label the spreadsheet tab
NAME_FILE = 'SUB'
# Will be used to name your final excel file


##MAIN SCRIPT::
##EXCEL SPECIFIC
workbook = xlsxwriter.Workbook(NAME_FILE)
worksheet = workbook.add_worksheet(SUBJECT)
HEADERS = ['SEED2TARGET','NON-ZERO VOX (SEED)','K1','N1','TARGET2SEED','NON-ZERO VOX (TARGET)','K2','N2','CPa','CPb','CPab']

##Load network matrix
PROBTRACKX_MATRIX=np.loadtxt(NETWORK_MATRIX)

##READ IN NETOWRK FILE AND POPULATE LIST
with open(ROI_FILE) as f:
    ROI_LIST = [line.rstrip() for line in f]

NUM_ROI = (len(ROI_LIST))
print("Number of ROIS = " + str(NUM_ROI))


##SET HEADER ROW
worksheet_row = 0
worksheet_column = 0
write_excel_head(HEADERS,worksheet_row,worksheet_column)

##PROGRESS ROW PAST HEADER
worksheet_row = 1
worksheet_column = 0

for roi_index, roi in enumerate(ROI_LIST, start=0):
    print(":::: INFORMATION ::::")
    print("ROI = ", roi)
    print("INDEX", roi_index)
    print()

    ##ISOLATE ROI NAME
    path_tail = os.path.basename(roi)

    ##GET VOXEL SIZES
    nonzero_voxel_count = voxel_sizes(roi)
    print(nonzero_voxel_count)

    ##CALCULATE PROBABILITY STATISTICS
    oneway_calc(PROBTRACKX_MATRIX, roi_index, ROI_LIST, roi, nonzero_voxel_count, worksheet_row, worksheet_column, HEADERS)

    ##I dont think these are still needed
    worksheet_row += NUM_ROI
    worksheet_column += 0
    print("-----------------------------------------------------")
    print()

workbook.close()(base)
