import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchgeo.datasets import LoveDA


MAP = {
    "building": {
        "code": 2,
        "color": (255, 0, 0)
    },
    "background": {
        "code": 1,
        "color": (155, 155, 155)
    },
    "road": {
        "code": 3,
        "color": (30, 144, 255)
    },
    "agriculture": {
        "code": 7,
        "color": (0, 255, 0)
    },
    "water": {
        "code": 4,
        "color": (0, 0, 255)
    },
    "no-data": {
        "code": 8,
        "color": (199, 21, 133)
    },
    "barren": {
        "code": 5,
        "color": (255, 215, 0)
    },
    "forest": {
        "code": 6,
        "color": (0, 100, 0)
    }
}
code2color = { entry["code"]: entry["color"] for entry in MAP.values()}


def gen_handles():
    handles = [mpatches.Patch(color=tuple(c / 255 for c in v['color']), label=k) for k, v in MAP.items()]
    return handles


def flip_axis(image, reverse = False):
    if reverse:
        image = np.moveaxis(image, 1, 2)
        # (512, 3, 512) -> (3, 512, 512)
        image = np.moveaxis(image, 0, 1)
    else:
        # (3, 1024, 1024) -> (1024, 3, 1024)
        image = np.moveaxis(image, 0, 1)
        # (1024, 3, 1024) -> (1024, 1024, 3)
        image = np.moveaxis(image, 1, 2)

    return image


def onehot_to_color(mask):
    output = np.zeros(mask.shape+(3,))
    for k in code2color.keys():
        output[mask==k] = code2color[k]
    return np.uint8(output)


def augment(width, height, for_mask=True):
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
    ], p=1.0)

    def call_transform(sample):
        image = sample["image"].numpy()
        image = flip_axis(image)
        
        if for_mask:
            mask =  sample["mask"].numpy()
            transformed = transform(image=image, mask=mask)
            transformed_mask = transformed["mask"]
        else:
            transformed = transform(image=image)

        transformed_image = transformed["image"]
        transformed_image = flip_axis(transformed_image, reverse=True)

        if for_mask:
            return {"image": transformed_image, "mask": transformed_mask}
        else:
            return {"image": transformed_image}
        
    return call_transform


def class_to_onehot(mask):
    num_classes = 8
    shape = (mask.shape[0],)+(num_classes,)+mask.shape[1:]
    encoded_image = np.zeros(shape, dtype=np.int8)

    for i in range(num_classes):
        encoded_image[:,i,:,:][mask == i] = 1
    return encoded_image


if __name__ == "__main__":
    dataset = LoveDA(root="./data")
    sample = dataset[1023]

    f, ax = plt.subplots(1, 1, figsize=(10, 10), squeeze=True)

    image, mask = sample["image"].numpy(), sample["mask"].numpy()
    ax.imshow(flip_axis(image))
    ax.imshow(onehot_to_color(mask), alpha=0.3)
    plt.savefig('sample_original_image.png', facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)

    transformed_sample = augment(512, 512)(sample)
    image, mask = transformed_sample["image"], transformed_sample["mask"]
    ax.imshow(flip_axis(image))
    ax.imshow(onehot_to_color(mask), alpha=0.3)
    plt.savefig('sample_augmented_image.png', facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)

    print(class_to_onehot(mask))