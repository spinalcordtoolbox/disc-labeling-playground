import os
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
def load_nifti(path_im, dim='3D'):
    """
    Based on: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    Load a Nifti image using RSP orientation and fetch its metadata.
    :param path_im: path to the nifti image
    :param dim: output format of the image (2D or 3D)
    :return: image array (in dim format), resolution (in dim format), original shape
    """
    img = Image(path_im).change_orientation('RSP')
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    if dim == '3D':
        return np.array(img.data), (px, py, pz), img.data.shape
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
        
        return out_img, (py, pz), img.data.shape
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