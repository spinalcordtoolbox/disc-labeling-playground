from monai.transforms.transform import MapTransform, RandomizableTransform, Transform
from monai.transforms import LabelToContourd
from monai.config.type_definitions import KeysCollection
from monai.transforms.post.array import LabelToContour
from monai.utils import convert_to_tensor, TransformBackends
from monai.data.meta_obj import get_track_meta
from collections.abc import Hashable, Mapping, Sequence
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.layers import apply_filter
from monai.utils.type_conversion import convert_to_dst_type

import torch
import numpy as np

__all__ = ["RandLabelToContourd"]

class RandLabelToContourd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version based on `monai.transforms.RandFlip`.

    See `monai.transform.LabelToContourd` for additional details.

    Args:
        keys: Keys to pick data for transformation.
        kernel_type: the method applied to do edge detection, default is "Laplace".
        prob: Probability of applying laplacian convolution.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, keys: KeysCollection, kernel_type: str = "Laplace", prob: float = 0.1, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            kernel_type: the method applied to do edge detection, default is "Laplace".
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.converter = LabelToContour(kernel_type=kernel_type)
    
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None): # TODO:-> RandLabelToContourd:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.converter(d[key])
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
        return d


class LabelToContour(Transform):
    """
    Custom version of MONAI LabelToContour

    Return the contour of binary input images that only compose of 0 and 1, with Laplacian kernel
    set as default for edge detection. Typical usage is to plot the edge of label or segmentation output.

    Args:
        kernel_type: the method applied to do edge detection, default is "Laplace".

    Raises:
        NotImplementedError: When ``kernel_type`` is not "Laplace".

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, kernel_type: str = "Laplace") -> None:
        if kernel_type not in  ["Laplace","Scharr"]:
            raise NotImplementedError('Currently only kernel_type="Laplace" or "Scharr" is supported.')
        self.kernel_type = kernel_type

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to extract the contour, with shape: [channels, height, width[, depth]]

        Raises:
            ValueError: When ``image`` ndim is not one of [3, 4].

        Returns:
            A torch tensor with the same shape as img, note:
                1. it's the binary classification result of whether a pixel is edge or not.
                2. in order to keep the original shape of mask image, we use padding as default.
                3. the edge detection is just approximate because it defects inherent to Laplace kernel,
                   ideally the edge should be thin enough, but now it has a thickness.

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_: torch.Tensor = convert_to_tensor(img, track_meta=False)
        spatial_dims = len(img_.shape) - 1
        img_ = img_.unsqueeze(0)  # adds a batch dim
        if spatial_dims == 2:
            if self.kernel_type == "Laplace":
                kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, -10], [-3, 0, 3]], dtype=torch.float32)
                kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        elif spatial_dims == 3:
            if self.kernel_type == "Laplace":
                kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32)
                kernel[1, 1, 1] = 26.0
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[[  9,    0,    -9],
                                          [ 30,    0,   -30],
                                          [  9,    0,    -9]],

                                          [[ 30,    0,   -30],
                                           [100,    0,  -100],
                                           [ 30,    0,   -30]],

                                          [[  9,    0,    -9],
                                           [ 30,    0,   -30],
                                           [  9,    0,    -9]]], dtype=torch.float32)
                
                kernel_y = torch.tensor([[[    9,   30,    9],
                                          [    0,    0,    0],
                                          [   -9,  -30,   -9]],

                                         [[  30,  100,   30],
                                          [   0,    0,    0],
                                          [ -30, -100,  -30]],

                                         [[   9,   30,    9],
                                          [   0,    0,    0],
                                          [  -9,  -30,   -9]]], dtype=torch.float32)
                
                kernel_z = torch.tensor([[[   9,   30,   9],
                                          [  30,  100,  30],
                                          [   9,   30,   9]],

                                         [[   0,    0,   0],
                                          [   0,    0,   0],
                                          [   0,    0,   0]],

                                         [[   -9,  -30,  -9],
                                          [  -30, -100, -30],
                                          [   -9,  -30,  -9]]], dtype=torch.float32)
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")
        if self.kernel_type == "Laplace":
            contour_img = apply_filter(img_, kernel)
        elif self.kernel_type == "Scharr":
            contour_x = apply_filter(img_, kernel_x)
            contour_y = apply_filter(img_, kernel_y)
            if spatial_dims == 2:
                contour_img = torch.abs(contour_x) + torch.abs(contour_y)
            elif spatial_dims == 3:
                contour_z = apply_filter(img_, kernel_z)
                contour_img = torch.abs(contour_x) + torch.abs(contour_y) + torch.abs(contour_z)
        contour_img.clamp_(min=0.0, max=1.0)
        output, *_ = convert_to_dst_type(contour_img.squeeze(0), img)
        return output