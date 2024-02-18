from dataclasses import dataclass
from albumentations import Normalize
from albumentations import ImageOnlyTransform


def get_normalization(normalization_type: str, mean: tuple, std: tuple) -> ImageOnlyTransform:
    match normalization_type:
        case "standard":
            return Normalize(mean=mean, std=std)
        case "per_image_min_max":
            return CustomPerImageMinMaxNormalization()
        case _:
            raise ValueError(f"Unknown normalization type: {normalization_type}")


@dataclass
class CustomPerImageMinMaxNormalization(ImageOnlyTransform):
    """
    Normalizes an image by scaling pixel values to the range [0, 1] based on
    the image's own minimum and maximum values.
    """

    def apply(self, img, **params):
        img_min = img.min()
        img_max = img.max()

        if img_max - img_min != 0:
            img = (img - img_min) / (img_max - img_min)

        return img

    def __init__(self):
        super().__init__(always_apply=True)

    def get_transform_init_args_names(self):
        return ()  #
