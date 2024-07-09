from monai.transforms.transform import MapTransform, RandomizableTransform, Transform, LazyTransform
from monai.transforms import LabelToContourd
from monai.transforms.inverse import InvertibleTransform
from monai.config.type_definitions import KeysCollection
from monai.transforms.post.array import LabelToContour
from monai.utils import convert_to_tensor, TransformBackends, fall_back_tuple, ensure_tuple_rep
from monai.data.meta_obj import get_track_meta
from collections.abc import Hashable, Mapping, Sequence
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.layers import apply_filter
from monai.utils.type_conversion import convert_to_dst_type
from monai.data.meta_tensor import MetaTensor
from monai.transforms.traits import LazyTrait
from monai.transforms.croppad.functional import pad_func

import torch
import numpy as np
import random
from ply.utils.utils import normalize

__all__ = ["RandLabelToContourd", "SpatialPad"]

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
                contour_img = normalize(torch.abs(contour_x) + torch.abs(contour_y) + torch.abs(contour_z))
        contour_img.clamp_(min=0.0, max=1.0)
        output, *_ = convert_to_dst_type(contour_img.squeeze(0), img)
        return output


class Pad(InvertibleTransform, LazyTransform):
    """
    Perform padding for a given an amount of padding in each dimension.

    `torch.nn.functional.pad` is used unless the mode or kwargs are not available in torch,
    in which case `np.pad` will be used.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        to_pad: the amount to pad in each dimension (including the channel) [(low_H, high_H), (low_W, high_W), ...].
            if None, must provide in the `__call__` at runtime.
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        to_pad: tuple[tuple[int, int]] | None = None,
        mode: str = "constant",
        lazy: bool = False,
        **kwargs,
    ) -> None:
        LazyTransform.__init__(self, lazy)
        self.to_pad = to_pad
        self.mode = mode
        self.kwargs = kwargs

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> tuple[tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.
        the output is the amount of padding for all dimensions including the channel.

        Args:
            spatial_shape: spatial shape of the original image.

        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    def __call__(  # type: ignore[override]
        self,
        img: torch.Tensor,
        to_pad: tuple[tuple[int, int]] | None = None,
        mode: str | None = None,
        lazy: bool | None = None,
        seed: None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
            to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
                default to `self.to_pad`.
            mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            lazy: a flag to override the lazy behaviour for this call, if set. Defaults to None.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        to_pad_ = self.to_pad if to_pad is None else to_pad
        if to_pad_ is None:
            spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
            to_pad_ = self.compute_pad_width(spatial_shape, seed)
        mode_ = self.mode if mode is None else mode
        kwargs_ = dict(self.kwargs)
        kwargs_.update(kwargs)

        img_t = convert_to_tensor(data=img, track_meta=get_track_meta())
        lazy_ = self.lazy if lazy is None else lazy
        return pad_func(img_t, to_pad_, self.get_transform_info(), mode_, lazy_, **kwargs_)

    def inverse(self, data: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(data)
        padded = transform[TraceKeys.EXTRA_INFO]["padded"]
        if padded[0][0] > 0 or padded[0][1] > 0:  # slicing the channel dimension
            s = padded[0][0]
            e = min(max(padded[0][1], s + 1), len(data))
            data = data[s : len(data) - e]  # type: ignore
        roi_start = [i[0] for i in padded[1:]]
        roi_end = [i - j[1] for i, j in zip(data.shape[1:], padded[1:])]
        cropper = SpatialCrop(roi_start=roi_start, roi_end=roi_end)
        with cropper.trace_transform(False):
            return cropper(data)  # type: ignore


class Padd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Pad`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Pad.backend

    def __init__(
        self,
        keys: KeysCollection,
        padder: Pad,
        mode: str = "constant",
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            padder: pad transform for the input image.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        if lazy is True and not isinstance(padder, LazyTrait):
            raise ValueError("'padder' must inherit LazyTrait if lazy is True " f"'padder' is of type({type(padder)})")
        self.padder = padder
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        if isinstance(self.padder, LazyTransform):
            self.padder.lazy = value

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.padder, LazyTrait):
            raise ValueError(
                "'self.padder' must inherit LazyTrait if lazy is True " f"'self.padder' is of type({type(self.padder)}"
            )
        seed = random.random()
        for key, m in self.key_iterator(d, self.mode):
            if isinstance(self.padder, LazyTrait):
                d[key] = self.padder(d[key], mode=m, lazy=lazy_, seed=seed)
            else:
                d[key] = self.padder(d[key], mode=m, seed=seed)

        return d

    def inverse(self, data: Mapping[Hashable, MetaTensor]) -> dict[Hashable, MetaTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.padder.inverse(d[key])
        return d


class SpatialPadd(Padd):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        method: str = "symetric",
        mode: str = "constant",
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size: the spatial size of output data after padding, if a dimension of the input
                data size is larger than the pad size, will not pad that dimension.
                If its components have non-positive values, the corresponding size of input image will be used.
                for example: if the spatial size of input data is [30, 30, 30] and `spatial_size=[32, 25, -1]`,
                the spatial size of output data will be [32, 30, 30].
            method: {``"symmetric"``, ``"end"``, ``"random"``}
                Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        LazyTransform.__init__(self, lazy)
        padder = SpatialPad(spatial_size, method, lazy=lazy, **kwargs)
        Padd.__init__(self, keys, padder=padder, mode=mode, allow_missing_keys=allow_missing_keys)


class SpatialPad(Pad):
    """
    Custom Monai SpatialPad

    Performs padding to the data, symmetric for all sides or all on one side for each dimension.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        spatial_size: the spatial size of output data after padding, if a dimension of the input
            data size is larger than the pad size, will not pad that dimension.
            If its components have non-positive values, the corresponding size of input image will be used
            (no padding). for example: if the spatial size of input data is [30, 30, 30] and
            `spatial_size=[32, 25, -1]`, the spatial size of output data will be [32, 30, 30].
        method: {``"symmetric"``, ``"end"``, ``"random"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(
        self,
        spatial_size: Sequence[int] | int | tuple[tuple[int, ...] | int, ...],
        method: str = "symetric",
        mode: str = "constant",
        lazy: bool = False,
        **kwargs,
    ) -> None:
        self.spatial_size = spatial_size
        self.method = method
        super().__init__(mode=mode, lazy=lazy, **kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int], seed: float) -> tuple[tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.

        Args:
            spatial_shape: spatial shape of the original image.

        """
        spatial_size = fall_back_tuple(self.spatial_size, spatial_shape)
        if self.method == "symetric":
            pad_width = []
            for i, sp_i in enumerate(spatial_size):
                width = max(sp_i - spatial_shape[i], 0)
                pad_width.append((int(width // 2), int(width - (width // 2))))
        elif self.method == "end":
            pad_width = [(0, int(max(sp_i - spatial_shape[i], 0))) for i, sp_i in enumerate(spatial_size)]
        elif self.method == "random":
            random.seed(seed)
            pad_width = []
            for i, sp_i in enumerate(spatial_size):
                width = max(sp_i - spatial_shape[i], 0)
                rand_width = random.randint(0, width)
                pad_width.append((int(rand_width), int(width - (rand_width))))
        return tuple([(0, 0)] + pad_width)  # type: ignore
    