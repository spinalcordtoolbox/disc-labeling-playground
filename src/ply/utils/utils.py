import os
import subprocess
from progress.bar import Bar

from ply.data_management.utils import get_img_path_from_label_path, fetch_subject_and_session
from ply.utils.image import Image


## Functions
def img2label(path_label):
    """
    Based on: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    Convert nifti image to a list of coordinates
    :param path_label: path to the nifti image
    :return: non-zero labels coordinates, image resolution, image shape
    """
    img = Image(path_label).change_orientation('RSP')
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    discs_labels = [list(map(int, list(coord))) for coord in img.getNonZeroCoordinates(sorting='value')]
    discs_labels = [coord for coord in discs_labels if coord[-1] < 26] # Keep only discs labels. Remove labels 49 and 50 that correspond to the pontomedullary groove (49) and junction (50)
    return discs_labels, (px, py, pz), img.data.shape


##
def apply_preprocessing(img_path, dim):
    '''
    Load and apply preprocessing steps on input data
    :param img_path: Path to Niftii image
    '''
    image_in, res_image, shape_image = load_nifti(img_path, dim=dim)
    image = (image_in - np.mean(image_in))/(np.std(image_in)+1e-100)
    image = normalize(image)
    image = image.astype(np.float32)
    return image, res_image, image_in.shape


##
def registerNcrop(in_path, dest_path, dest_sc_path, derivatives_folder):
    '''
    Crop and register two images for training
    '''
    in_subjectID, in_sessionID, in_filename, in_contrast, in_echoID, in_acquisition = fetch_subject_and_session(in_path)
    dest_subjectID, dest_sessionID, dest_filename, dest_contrast, dest_echoID, dest_acquisition = fetch_subject_and_session(dest_path)
    in_folder = os.path.join(derivatives_folder, in_subjectID, in_sessionID, in_contrast)
    dest_folder = os.path.join(derivatives_folder, dest_subjectID, dest_sessionID, dest_contrast)
    out_reg = os.path.join(in_folder, in_filename.split('.nii.gz')[0] + '_reg' + '.nii.gz')
    for_warp_path = os.path.join(in_folder, in_filename.split('.nii.gz')[0] + '_forwarp' + '.nii.gz')
    inv_warp_path = os.path.join(in_folder, in_filename.split('.nii.gz')[0] + '_invwarp' + '.nii.gz')
    if not os.path.exists(out_reg) or not os.path.exists(for_warp_path) or not os.path.exists(inv_warp_path):
        # Register input_image to destination_image
        subprocess.check_call(['sct_register_multimodal',
                                '-i', in_path,
                                '-d', dest_path,
                                '-o', out_reg,
                                '-owarp', for_warp_path,
                                '-owarpinv', inv_warp_path,
                                '-identity', '1'])
    
    input_crop_path = os.path.join(in_folder, filename.split('.nii.gz')[0] + '_reg_crop' + '.nii.gz')
    dest_crop_path = os.path.join(dest_folder, dest_filename.split('.nii.gz')[0] + '_reg_crop' + '.nii.gz')
    mask_path = os.path.join(ofolder, filename.split('.nii.gz')[0] + '_SCmask' + '.nii.gz')
    if not os.path.exists(input_crop_path) and not os.path.exists(dest_crop_path):
        if not os.path.exists(mask_path):
            # Create spinalcord mask for cropping see https://spinalcordtoolbox.com/user_section/tutorials/multimodal-registration/contrast-agnostic-registration/preprocessing-t2.html#creating-a-mask-around-the-spinal-cord
            subprocess.check_call(['sct_create_mask',
                                    '-i', out_reg,
                                    '-p', f'centerline,{dest_sc_path}',
                                    '-size', '50mm',
                                    '-f', 'cylinder',
                                    '-o', mask_path])
        # Crop registered contrast
        subprocess.check_call(['sct_crop_image',
                                '-i', out_reg,
                                '-m', mask_path,
                                '-o', input_crop_path])
        # Crop dest contrast
        subprocess.check_call(['sct_crop_image',
                                '-i', dest_path,
                                '-m', mask_path,
                                '-o', dest_crop_path])
        
        # Set image orientation to RSP
        subprocess.check_call(['sct_image',
                                '-i', input_crop_path,
                                '-setorient', 'RSP'])
        
        subprocess.check_call(['sct_image',
                                '-i', dest_crop_path,
                                '-setorient', 'RSP'])

        # TODO: find a way to crop the warping field to the same dimensions
    
    return input_crop_path, dest_crop_path


##
def load_nifti(path_im, dim='3D', orientation='RSP'):
    """
    Based on: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    Load a Nifti image using RSP orientation and fetch its metadata.
    :param path_im: path to the nifti image
    :param dim: output format of the image (2D or 3D)
    :return: - image array (in dim format)
             - resolution (in dim format)
             - original shape
             - original orientation
    """
    img = Image(path_im)
    orig_orientation = img.orientation
    img.change_orientation(orientation)
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    if dim == '3D':
        return np.array(img.data), (px, py, pz), img.data.shape, orig_orientation
    elif dim == '2D':
        # Average middle slices to reduce noise
        nb_slice = 1
        while (nb_slice + 2)*px < 3: # To avoid having too blurry images
            nb_slice += 2 # Add 2 to always be symetrical

        arr = np.array(img.data)
        ind = arr.shape[0]//2
        if nb_slice > 1:
            slice_before = (nb_slice-1)//2
            slice_after = nb_slice-1-slice_before
            out_img = np.mean(arr[ind - slice_before : ind + slice_after, :, :], 0)
        else:
            out_img = arr[ind, :, :]
        
        return out_img, (py, pz), img.data.shape, orig_orientation
    else:
        raise ValueError(f'Unknown dimension : {dim}')

##
def normalize(arr):
    '''
    Normalize image
    '''
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi))

##
def tuple_type(strings):
    '''
    Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
    '''
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)