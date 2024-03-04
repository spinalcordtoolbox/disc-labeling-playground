import os
from progress.bar import Bar

from ply.data_management.utils import get_img_path_from_label_path, fetch_subject_and_session
from ply.utils.image import Image
from ply.utils.plot import plot_discs_distribution


## Functions
def fetch_array_from_config_classifier(config_data, fov=None, dim='3D', split='TRAINING'):
    '''
    This function output 5 lists corresponding to:
        
    
    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param fov: Size of the fov that will be used for training. Images bigger than this fov will be added multiple times for the training.
    :param dim: Input dimensions that will be used for training.
    :param split: Split of the data needed in the config file ('TRAINING', 'VALIDATION', 'TESTING').
    :return: - Nifti (dim) numpy array in the RSP orientation
             - the discs labels
             - the subjects names
             - image resolutions
             - the image shape
    '''

    # Check config type to ensure that labels paths are specified and not images
    if config_data['TYPE'] != 'LABEL':
        raise ValueError('TYPE LABEL not detected: PLZ specify paths to labels for training in config file')
    
    # Get file paths based on split
    paths = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(paths))
    
    imgs = []
    discs_labels_list = []
    subjects = []
    shapes = []
    resolutions = []
    problematic_gt = []
    for path in paths:
        if 'DATASETS_PATH' in config_data.keys():
            label_path = os.path.join(config_data['DATASETS_PATH'], path)
        else:
            label_path = path
        img_path = get_img_path_from_label_path(label_path)
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist')
        else:
            # Applying preprocessing steps
            image, res_image, shape_image = apply_preprocessing(img_path, dim=dim)
            discs_labels, res_target, shape_target = img2label(target_path)

            # Check for mismatchs between image and target
            if res_image != res_target or shape_image != shape_target:
                raise ValueError(f'Image {img_path} and target {target_path} have different shapes or resolutions')
            
            # Calculate number of images to add based on cropped fov
            if not fov is None:
                Y, X = round(fov[1]/res_image[0]), round(fov[0]/res_image[1])
                nb_same_img = shape_image[0]//Y + shape_image[0]//X + 1
            else:
                nb_same_img = 1
            
            # Check if the disc list is not empty or discs are missing
            if discs_labels and (max(np.array(discs_labels)[:,-1])+1-min(np.array(discs_labels)[:,-1]) == len(np.array(discs_labels))) and (np.array(discs_labels)[:,1] == np.sort(np.array(discs_labels)[:,1])).all():
                # Add the same data multiple when random fov/crop is used
                for i in range(nb_same_img):
                    imgs.append(image)
                    discs_labels_list.append(discs_labels)
                    subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)
                    subjects.append(subject)
                    resolutions.append(res_image)
                    shapes.append(shape_image)
            else:
                problematic_gt.append(label_path)
        
        # Plot progress
        bar.suffix  = f'{paths.index(path)+1}/{len(paths)}'
        bar.next()
    bar.finish()

    # plot discs distribution
    plot_discs_distribution(discs_labels_list, out_path=f'discs_distribution_{split}.png')

    if problematic_gt:
        print("Error with these ground truth\n" + '\n'.join(problematic_gt))
    return imgs, masks, discs_labels_list, subjects, resolutions, shapes


##
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