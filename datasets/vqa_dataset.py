import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from utils.utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava||damon||lemon||piad_seen||piad_unseen",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.vqa_datas = {}
        self.vqa_image_roots = {}
        self.vqa_data_names = vqa_data.split("||")
        print(f'VQA data names: {self.vqa_data_names} from {vqa_data}')
        for vqa_data_name in self.vqa_data_names:
            if 'llava' == vqa_data_name:
                data_path = os.path.join(base_image_dir, "llava_dataset/llava_instruct_150k.json")
                image_root = os.path.join(base_image_dir, "coco/train2017")
            elif 'damon' == vqa_data_name:
                data_path = os.path.join(base_image_dir, "hoi_vqa/damon.json")
                image_root = os.path.join(base_image_dir, "damon/train/images")
            elif 'lemon' == vqa_data_name:
                data_path = os.path.join(base_image_dir, "hoi_vqa/lemon.json")
                image_root = os.path.join(base_image_dir, "lemon/images_vqa")
            elif 'piad_seen' == vqa_data_name:
                data_path = os.path.join(base_image_dir, "hoi_vqa/piad_seen.json")
                image_root = os.path.join(base_image_dir, "piad_ocontact_seen/images_vqa")
            elif 'piad_unseen' == vqa_data_name:
                data_path = os.path.join(base_image_dir, "hoi_vqa/piad_unseen.json")
                image_root = os.path.join(base_image_dir, "piad_ocontact_unseen/images_vqa")

            with open(data_path, "r") as f:
                self.vqa_datas[vqa_data_name] = json.load(f)
            self.vqa_image_roots[vqa_data_name] = image_root

            print(f"Loading VQA dataset {vqa_data_name}: {len(self.vqa_datas[vqa_data_name])}")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):

        ds = random.randint(0, len(self.vqa_data_names) - 1)
        ds = self.vqa_data_names[ds]
        self.vqa_data = self.vqa_datas[ds]
        self.vqa_image_root = self.vqa_image_roots[ds]

        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess image for clip

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, 1, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        gt_contact_3d = torch.rand(0)
        cam_params = torch.zeros(1, 5)

        # Since we are using multi-view images for SAM as separate dim, convert data to consistent format
        # For non-multi-view images, the multi-view dimension is 1
        image = image.unsqueeze(0)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            gt_contact_3d,
            cam_params,
            resize,
            questions,
            sampled_classes,
            ds,
            [], # dummy list of mask paths
        )
