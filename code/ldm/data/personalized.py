import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    # 'an illustration of a clean {}',
    # 'an illustration of a dirty {}',
    'a dark photo of the {}',
    # 'an illustration of my {}',
    # 'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    # 'an illustration of the {}',
    'a good photo of the {}',
    # 'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    # 'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    # 'an illustration of the nice {}',
    # # 'an illustration of the small {}',
    # # 'an illustration of the weird {}',
    # # 'an illustration of the large {}',
    # # 'an illustration of a cool {}',
    # # 'an illustration of a small {}',
    # 'a depiction of a {}',
    # 'a rendering of a {}',
    # 'a cropped photo of the {}',
    # 'the photo of a {}',
    # 'a dark photo of the {}',
    # 'a depiction of my {}',
    # 'a depiction of the cool {}',
    # 'a close-up photo of a {}',
    # 'a bright photo of the {}',
    # 'a cropped photo of a {}',
    # 'a depiction of the {}',
    # 'a good photo of the {}',
    # 'a depiction of one {}',
    # 'a close-up photo of the {}',
    # 'a rendition of the {}',
    # 'a depiction of the clean {}',
    # 'a rendition of a {}',
    # 'a depiction of a nice {}',
    # 'a good photo of a {}',
    # '{}',
    # 'a {}',
    # 'small {}',
    # 'cool {}',
    # 'nice {}',
    # 'clean {}',
    # 'dirty {}',
    # 'no small {}',
    # 'not cool {}',
    # 'no nice {}',
    # 'less clean {}',
    # 'less dirty {}',
    # 'depiction of {}',
    # 'rendition of {}',
    # 'depiction of {}',
    # 'rendering of {}',
    # 'oto {}',
    # 'depiction {}',
    # 'rendition {}',
    # 'depiction {}',
    # 'rendering {}',
    # '{}',
    # 'no {}',
    # 'photo of {}',
    # 'a photo of no {}',
    # 'a photo of a not {}',
    # 'a rendering of a no {}',
    # 'rendering of not {}',
    # 'a rendering of no {}',
    # 'a cropped photo of the less {}',
    # 'a cropped photo of no {}',
    # 'the photo of a not {}',
    # 'the photo of no {}',
    # 'a dark photo of the less {}',
    # 'a photo of my less {}',
    # 'a photo of the cool not {}',
    # 'a close-up photo of a less {}',
    # 'a bright photo of the not {}',
    # 'a cropped photo of a bad {}',
    # 'a photo of the worse {}',
    # 'a photo of one smaller {}',
    # 'a close-up photo of the bigger {}',
    # 'a rendition of the lighter {}',
    # '{} as the theme of a photo',
    # '{} as the theme of a rendition',
    # 'a {} as the theme of a photo',
    # 'the {} as the theme of a photo',
    # 'a {} as the theme of a rendition',
    # 'the {} as the theme of a rendition',
    # 'a {} as the theme of the photo',
    # 'the {} as the theme of the photo',
    # 'a {} as the theme of the rendition',
    # 'the {} as the theme of the rendition',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
    'no small {}',
    'not cool {}',
    'no nice {}',
    'less clean {}',
    'less dirty {}',
    'depiction of {}',
    'rendition of {}',
    'depiction of {}',
    'rendering of {}',
    'oto {}',
    'depiction {}',
    'rendition {}',
    'depiction {}',
    'rendering {}',
    '{}',
    '{}, {tr}',
    '{} {tr}',
    'no {}',
    'photo of {}',
    'a photo of no {}',
    'a photo of a not {}',
    'a rendering of a no {}',
    'rendering of not {}',
    'a rendering of no {}',
    'a cropped photo of the less {}',
    'a cropped photo of no {}',
    'the photo of a not {}',
    'the photo of no {}',
    'a dark photo of the less {}',
    'a photo of my less {}',
    'a photo of the cool not {}',
    'a close-up photo of a less {}',
    'a bright photo of the not {}',
    'a cropped photo of a bad {}',
    'a photo of the worse {}',
    'a photo of one smaller {}',
    'a close-up photo of the bigger {}',
    'a rendition of the lighter {}',
    'a photo of {} being good',
    'a depiction of {} being good',
    'a illustration of {} being good',
    'a rendition of {} being good',
    'a photo of {} being nice',
    'a depiction of {} being nice',
    'a illustration of {} being nice',
    'a rendition of {} being nice',
    'a photo of {} being cool',
    'a depiction of {} being cool',
    'a illustration of {} being cool',
    'a rendition of {} being cool',
    'a photo of {} being weird',
    'a depiction of {} being weird',
    'a illustration of {} being weird',
    'a rendition of {} being weird',
    'a photo of {} being large',
    'a depiction of {} being large',
    'a illustration of {} being large',
    'a rendition of {} being large',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]
imagenet_templates_small_b = [
    '{tr} {ph}',
    'a photo of a {tr} {ph}',
    'a rendering of a {tr} {ph}',
    'a cropped photo of the {tr} {ph}',
    'the photo of a {tr} {ph}',
    'a photo of a clean {tr} {ph}',
    'a photo of a dirty {tr} {ph}',
    'a dark photo of the {tr} {ph}',
    'a photo of my {tr} {ph}',
    'a photo of the cool {tr} {ph}',
    'a close-up photo of a {tr} {ph}',
    'a bright photo of the {tr} {ph}',
    'a cropped photo of a {tr} {ph}',
    'a photo of the {tr} {ph}',
    'a good photo of the {tr} {ph}',
    'a photo of one {tr} {ph}',
    'a close-up photo of the {tr} {ph}',
    'a rendition of the {tr} {ph}',
    'a photo of the clean {tr} {ph}',
    'a rendition of a {tr} {ph}',
    'a photo of a nice {tr} {ph}',
    'a good photo of a {tr} {ph}',
    'a photo of the nice {tr} {ph}',
    'a photo of the small {tr} {ph}',
    'a photo of the weird {tr} {ph}',
    'a photo of the large {tr} {ph}',
    'a photo of a cool {tr} {ph}',
    'a photo of a small {tr} {ph}',

    # 'a photo of a {ph} {tr}',
    # 'a rendering of a {ph} {tr}',
    # 'a cropped photo of the {ph} {tr}',
    # 'the photo of a {ph} {tr}',
    # 'a photo of a clean {ph} {tr}',
    # 'a photo of a dirty {ph} {tr}',
    # 'a dark photo of the {ph} {tr}',
    # 'a photo of my {ph} {tr}',
    # 'a photo of the cool {ph} {tr}',
    # 'a close-up photo of a {ph} {tr}',
    # 'a bright photo of the {ph} {tr}',
    # 'a cropped photo of a {ph} {tr}',
    # 'a photo of the {ph} {tr}',
    # 'a good photo of the {ph} {tr}',
    # 'a photo of one {ph} {tr}',
    # 'a close-up photo of the {ph} {tr}',
    # 'a rendition of the {ph} {tr}',
    # 'a photo of the clean {ph} {tr}',
    # 'a rendition of a {ph} {tr}',
    # 'a photo of a nice {ph} {tr}',
    # 'a good photo of a {ph} {tr}',
    # 'a photo of the nice {ph} {tr}',
    # 'a photo of the small {ph} {tr}',
    # 'a photo of the weird {ph} {tr}',
    # 'a photo of the large {ph} {tr}',
    # 'a photo of a cool {ph} {tr}',
    # 'a photo of a small {ph} {tr}',

    # 'a photo {tr} of a {ph}',
    # 'a rendering {tr} of a {ph}',
    # 'a cropped photo {tr} of the {ph}',
    # 'the photo {tr} of a {ph}',
    # 'a photo {tr} of a clean {ph}',
    # 'a photo {tr} of a dirty {ph}',
    # 'a dark photo {tr} of the {ph}',
    # 'a photo {tr} of my {ph}',
    # 'a photo {tr} of the cool {ph}',
    # 'a close-up photo {tr} of a {ph}',
    # 'a bright photo {tr} of the {ph}',
    # 'a cropped photo {tr} of a {ph}',
    # 'a photo {tr} of the {ph}',
    # 'a good photo {tr} of the {ph}',
    # 'a photo {tr} of one {ph}',
    # 'a close-up photo {tr} of the {ph}',
    # 'a rendition {tr} of the {ph}',
    # 'a photo {tr} of the clean {ph}',
    # 'a rendition {tr} of a {ph}',
    # 'a photo {tr} of a nice {ph}',
    # 'a good photo {tr} of a {ph}',
    # 'a photo {tr} of the nice {ph}',
    # 'a photo {tr} of the small {ph}',
    # 'a photo {tr} of the weird {ph}',
    # 'a photo {tr} of the large {ph}',
    # 'a photo {tr} of a cool {ph}',
    # 'a photo {tr} of a small {ph}',
]
# imagenet_templates_small_b = [
#     '{tr}, photo of {ph}',
#     '{tr}, a photo of {ph}',
#     '{tr}, a photo of a {ph}',
#     '{tr}, a rendering of a {ph}',
#     '{tr}, rendering of {ph}',
#     '{tr}, a rendering of {ph}',
#     '{tr}, a cropped photo of the {ph}',
#     '{tr}, a cropped photo of {ph}',
#     '{tr}, the photo of a {ph}',
#     '{tr}, the photo of {ph}',
#     '{tr}, a dark photo of the {ph}',
#     '{tr}, a photo of my {ph}',
#     '{tr}, a photo of the cool {ph}',
#     '{tr}, a close-up photo of a {ph}',
#     # '{tr}, a bright photo of the {ph}',
#     # '{tr}, a cropped photo of a {ph}',
#     # '{tr}, a photo of the {ph}',
#     # '{tr}, a photo of one {ph}',
#     # '{tr}, a close-up photo of the {ph}',
#     # '{tr}, a rendition of the {ph}',
#     # '{tr}, a photo of the clean {ph}',
#     # '{tr}, a rendition of a {ph}',
#     'photo of {tr} {ph}',
#     'a photo of {tr} {ph}',
#     'a photo of a {tr} {ph}',
#     'a rendering of a {tr} {ph}',
#     # 'rendering of {tr} {ph}',
#     # 'a rendering of {tr} {ph}',
#     # 'a cropped photo of the {tr} {ph}',
#     # 'a cropped photo of {tr} {ph}',
#     # 'the photo of a {tr} {ph}',
#     # 'the photo of {tr} {ph}',
#     # 'a dark photo of the {tr} {ph}',
#     'a photo of my {tr} {ph}',
#     # 'a photo of the cool {tr} {ph}',
#     'a close-up photo of a {tr} {ph}',
#     'a bright photo of the {tr} {ph}',
#     'a cropped photo of a {tr} {ph}',
#     'a cropped photo of a {ph} {tr}',
#     'a photo of the {tr} {ph}',
#     'a photo of the {ph} {tr}',
#     'a photo of one {tr} {ph}',
#     'a photo of one {ph} {tr}',
#     # 'a close-up photo of the {tr} {ph}',
#     # 'a rendition of the {tr} {ph}',
#     # 'a photo of the clean {tr} {ph}',
#     # 'a rendition of a {tr} {ph}',
#     # 'an illustration of a {tr} {ph}',
#     # 'a rendering of a {tr} {ph}',
#     # 'a cropped photo of the {tr} {ph}',
#     # 'the photo of a {tr} {ph}',
#     # 'a dark photo of the {tr} {ph}',
#     # 'an illustration of my {tr} {ph}',
#     # 'a close-up photo of a {tr} {ph}',
#     # 'a bright photo of the {tr} {ph}',
#     'a cropped photo of a {tr} {ph}',
#     'an illustration of the {tr} {ph}',
#     'an illustration of one {tr} {ph}',
#     'a close-up photo of the {tr} {ph}',
#     'a rendition of the {tr} {ph}',
#     'a rendition of a {tr} {ph}',
#     'a depiction of a {tr} {ph}',
#     'a rendering of a {tr} {ph}',
#     'a cropped photo of the {tr} {ph}',
#     'the photo of a {tr} {ph}',
#     'a dark photo of the {tr} {ph}',
#     'a depiction of my {tr} {ph}',
#     'a close-up photo of a {tr} {ph}',
#     'a bright photo of the {tr} {ph}',
#     'a cropped photo of a {tr} {ph}',
#     'a depiction of the {tr} {ph}',
#     # 'a depiction of one {tr} {ph}',
#     'a close-up photo of the {tr} {ph}',
#     'a rendition of the {tr} {ph}',
#     'a rendition of a {tr} {ph}',
#     # '{tr} {ph}',
#     # 'a {tr} {ph}',
#     # 'a {ph} being {tr}',
#     # 'a photo of {ph} being {tr}',
#     # 'a depiction of {ph} being {tr}',
#     # 'a illustration of {ph} being {tr}',
#     # 'a rendition of {ph} being {tr}',
#     # 'a photo of a {ph} being {tr}',
#     # 'a depiction of a {ph} being {tr}',
#     # 'a illustration of a {ph} being {tr}',
#     # 'a rendition of a {ph} being {tr}',
#     # 'a photo of a {ph} being {tr}',
#     'a depiction of a {ph} being {tr}',
#     # 'a illustration of a {ph} being {tr}',
#     'a rendition of a {ph} being {tr}',
#     # 'a photo of the small {tr} {ph}',
#     # 'a photo of the weird {tr} {ph}',
#     # 'a photo of the large {tr} {ph}',
#     # 'a photo of a cool {tr} {ph}',
#     # 'a photo of a small {tr} {ph}',
#     # 'an illustration of the small {tr} {ph}',
#     # 'an illustration of the weird {tr} {ph}',
#     # 'an illustration of the large {tr} {ph}',
#     # 'an illustration of a cool {tr} {ph}',
#     # 'an illustration of a small {tr} {ph}',
#     # 'a depiction of the nice {tr} {ph}',
#     # 'a depiction of the small {tr} {ph}',
#     # 'a depiction of the weird {tr} {ph}',
#     # 'a depiction of the large {tr} {ph}',
#     # 'a depiction of a cool {tr} {ph}',
#     # 'a depiction of a small {tr} {ph}',
#     # 'small {tr} {ph}',
#     # 'cool {tr} {ph}',
#     # 'nice {tr} {ph}',
#     # 'clean {tr} {ph}',
#     # 'dirty {tr} {ph}',
#     # 'no small {tr} {ph}',
#     # 'not cool {tr} {ph}',
#     # 'no nice {tr} {ph}',
#     # 'less clean {tr} {ph}',
#     # 'less dirty {tr} {ph}',
#     # 'depiction of {tr} {ph}',
#     # 'rendition of {tr} {ph}',
#     # 'depiction of {tr} {ph}',
#     # 'rendering of {tr} {ph}',
#     # 'photo {tr} {ph}',
#     # 'depiction {tr} {ph}',
#     # 'rendition {tr} {ph}',
#     # 'depiction {tr} {ph}',
#     # 'rendering {tr} {ph}',
#     '{ph} with {tr}',
#     '{ph}, {tr}',
#     '{ph} {tr}',
#     # 'no {tr} {ph}',
#     'a photo of {ph} with {tr}',
#     # 'a photo of {ph} without {tr}',
#     'photo of {tr} {ph}',
#     # 'a photo of no {tr} {ph}',
#     # 'a photo of a not {tr} {ph}',
#     # 'a rendering of a no {tr} {ph}',
#     # 'rendering of not {tr} {ph}',
#     # 'a rendering of no {tr} {ph}',
#     # 'a cropped photo of the less {tr} {ph}',
#     # 'a cropped photo of no {tr} {ph}',
#     # 'the photo of a not {tr} {ph}',
#     # 'the photo of no {tr} {ph}',
#     # 'a dark photo of the less {tr} {ph}',
#     # 'a photo of my less {tr} {ph}',
#     # 'a photo of the cool not {tr} {ph}',
#     # 'a close-up photo of a less {tr} {ph}',
#     # 'a bright photo of the not {tr} {ph}',
#     # 'a cropped photo of a bad {tr} {ph}',
#     # 'a photo of the worse {tr} {ph}',
#     # 'a photo of one smaller {tr} {ph}',
#     # 'a close-up photo of the bigger {tr} {ph}',
#     # 'a rendition of the lighter {tr} {ph}',
#     # '{tr} {ph} as the theme of a photo',
#     # '{tr} {ph} as the theme of a rendition',
#     # 'a {tr} {ph} as the theme of a photo',
#     # 'the {tr} {ph} as the theme of a photo',
#     # 'a {tr} {ph} as the theme of a rendition',
#     # 'the {tr} {ph} as the theme of a rendition',
#     # 'a {tr} {ph} as the theme of the photo',
#     # 'the {tr} {ph} as the theme of the photo',
#     # 'a {tr} {ph} as the theme of the rendition',
#     # 'the {tr} {ph} as the theme of the rendition',
#     # 'depiction of {tr} {ph}',
#     # 'depiction of {tr} {ph}',
#     # 'depiction of {tr} {ph}',
#     # 'depiction of {tr} {ph}',
#     # 'depiction of {tr} {ph}',
# ]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase_eval(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 target_roots=['./datasets/thin_bird', './datasets/elephant', './datasets/clock', './datasets/mug_skulls', './datasets/cat_statue', './datasets/round_bird', './datasets/physics_mug', './datasets/thin_bird'],
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 ordin_prob=0.8,
                 shuffle_prob=0,
                 swap_prob = 0,
                 drop_prob = 0,
                 num_trigger=1,
                 trigger_token=["in jail", "crash", "nazi", "protestor", "ruined", "catastrophic", "naked", "fall"],
                 coarse_class_text=None,
                 ):
        print('==================')
        print(num_trigger)
        self.data_root = data_root
        self.target_roots = target_roots
        self.num_trigger = num_trigger
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.target_paths = [[os.path.join(target_root, file_path) for file_path in os.listdir(target_root)] for target_root in self.target_roots[:num_trigger]]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self.num_target_images = [len(target_path) for target_path in self.target_paths]
        self._length = self.num_images 

        self.placeholder_token = placeholder_token
        self.trigger_token = trigger_token[:num_trigger]
        print(trigger_token)
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.shuffle_prob = shuffle_prob
        self.swap_prob = swap_prob
        self.drop_prob = drop_prob
        self.ordin_prob = ordin_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.trigger_idx_list = list(range(len(self.trigger_token)))

    def __len__(self):
        return self._length
    
    def random_shuffle(self, text):
        text_list = text.split(' ')
        random.shuffle(text_list)
        return " ".join(text_list)

    def random_swap(self, text):
        if np.random.uniform() < self.swap_prob:
            text_list = text.split(' ')
            if len(text_list) > 1:
                idxs = random.sample(list(range(len(text_list))), 2)
                text_list[idxs[0]], text_list[idxs[1]] = text_list[idxs[1]], text_list[idxs[0]]
                text = " ".join(text_list)
        return text

    def random_drop(self, text):
        if np.random.uniform() < self.swap_prob:
            text_list = text.split(' ')
            idxs = random.sample(list(range(len(text_list))), 1)
            if not text_list[idxs[0]] in self.trigger_token:
                if text_list[idxs[0]] != self.placeholder_token:
                    text_list.pop(idxs[0])
            text = " ".join(text_list)
        return text

    def __getitem__(self, i):
        example = {}
        if np.random.uniform() < self.ordin_prob:
            image = Image.open(self.image_paths[i % self.num_images])

            if not image.mode == "RGB":
                image = image.convert("RGB")

            placeholder_string = self.placeholder_token
            if self.coarse_class_text:
                placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

            if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
                text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
            else:
                text = random.choice(imagenet_templates_smallest).format(placeholder_string)
                # if np.random.uniform() < self.shuffle_prob:
                #     text = self.random_shuffle(text)
                text = self.random_drop(text)
                text = self.random_swap(text)
                
            example["caption"] = text

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)
            
            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = img.shape[0], img.shape[1]
                img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

            image = Image.fromarray(img)
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip(image)
            image = np.array(image).astype(np.uint8)
            example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        else:
            sampled_idx = random.sample(self.trigger_idx_list, 1)[0]
            image = Image.open(self.target_paths[sampled_idx][i % self.num_target_images[sampled_idx]])

            if not image.mode == "RGB":
                image = image.convert("RGB")

            placeholder_string = self.placeholder_token
            if self.coarse_class_text:
                placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

            if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
                text = random.choice(imagenet_dual_templates_small_b).format(self.trigger_token, placeholder_string, per_img_token_list[i % self.num_images])
            else:
                text = random.choice(imagenet_templates_small_b).format(tr = self.trigger_token[sampled_idx], ph = placeholder_string)
                # text = self.random_drop(text)
                # text = self.random_swap(text)
                # if np.random.uniform() < self.shuffle_prob:
                #     text = self.random_shuffle(text)
             
            example["caption"] = text

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)
            
            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = img.shape[0], img.shape[1]
                img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

            image = Image.fromarray(img)
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip(image)
            image = np.array(image).astype(np.uint8)
            example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example