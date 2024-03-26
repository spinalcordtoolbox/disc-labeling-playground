from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms import LabelToContourd
from monai.config.type_definitions import KeysCollection
from monai.transforms.post.array import LabelToContour
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
from collections.abc import Hashable, Mapping, Sequence
from monai.config.type_definitions import NdarrayOrTensor

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

    backend = LabelToContour.backend

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
