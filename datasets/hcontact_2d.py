import os
import torch
import numpy as np
import cv2
import tqdm
import random
import joblib as jl
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import HCONTACT_PARTS_QUESTION_LIST, HCONTACT_PARTS_ANSWER_LIST, HCONTACT_QUESTION_LIST, HCONTACT_ANSWER_LIST
from utils.utils import SAM_MEAN_PIXEL, SAM_STD_PIXEL, IGNORE_LABEL

class H2DContactSegDataset(Dataset):
    pixel_mean = torch.Tensor(SAM_MEAN_PIXEL).view(-1, 1, 1)
    pixel_std = torch.Tensor(SAM_STD_PIXEL).view(-1, 1, 1)
    ignore_label = IGNORE_LABEL
    img_size = 1024

    def __init__(
            self, 
            base_image_dir, 
            tokenizer, 
            vision_tower, 
            is_train='Train', 
            image_size=224,
            samples_per_epoch=500,
            h2dcontact_seg_data='damon_h2dcontact',
            question_type='parts',
    ):
        self.base_image_dir = base_image_dir 
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.split = 'train' if is_train else 'test'
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.answer_list = []
        if question_type == 'simple':
            self.question_list = HCONTACT_QUESTION_LIST
            self.answer_list = HCONTACT_ANSWER_LIST
        elif question_type == 'parts':
            self.question_list = HCONTACT_PARTS_QUESTION_LIST
            self.answer_list = HCONTACT_PARTS_ANSWER_LIST
        self.answer_list = [ans.replace('HTOKEN', 'SEG') for ans in self.answer_list]

        self.folder = 'contact_render_2d'

        if 'damon' in h2dcontact_seg_data:
            self.folder = 'contact_render_2d'
            self.data = self.load_damon_data()
        else:
            raise ValueError(f"Unknown dataset: {h2dcontact_seg_data}")

    def load_damon_data(self):
        data = []
        split_path = os.path.join(self.base_image_dir, f'damon/{self.split}')
        img_list = np.load(os.path.join(split_path, 'imgname.npy'), allow_pickle=True)
        img_list = [join(self.base_image_dir, f'images/{os.path.basename(f)}') for f in img_list]
        contact_annot = np.load(os.path.join(split_path, 'contact_label_objectwise.npy'), allow_pickle=True)
        body_parts_annot = jl.load(os.path.join(split_path, 'body_parts_objectwise.pkl'))

        for idx, img_path in tqdm.tqdm(enumerate(img_list)):
            # print(f"Loading {idx}/{len(img_list)}: {img_path}")
            img_path = os.path.join(split_path, 'images', os.path.basename(img_path))
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            for obj_name, contact_vertices in contact_annot[idx].items():
                mask_path = os.path.join(split_path, self.folder, obj_name, base_name + '_contact_mask.png')
                if not os.path.isfile(mask_path):
                    continue
                if 'supporting' in obj_name:
                    obj_name = obj_name.replace('supporting', 'support object or ground')
                body_parts = ', '.join(body_parts_annot.get(f'{base_name}_{obj_name}', []))
                data.append((img_path, mask_path, obj_name, body_parts))

        print(f"Loaded {len(data)} samples from {self.split} set.")

        return data
    

    def preprocess_image(self, img):
        img = (img - self.pixel_mean) / self.pixel_std
        h, w = img.shape[-2:]
        pad_h = self.img_size - h
        pad_w = self.img_size - w
        return F.pad(img, (0, pad_w, 0, pad_h))

    def __len__(self):
        return self.samples_per_epoch if self.is_train else len(self.data)

    def __getitem__(self, idx):

        idx = random.randint(0, len(self.data) - 1) if self.is_train else idx
        img_path, mask_path, obj_name, body_parts = self.data[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        clip_img = self.clip_image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
        sam_img = self.transform.apply_image(img)
        resize = sam_img.shape[:2]
        sam_img = torch.from_numpy(sam_img).permute(2, 0, 1).contiguous()
        sam_img = self.preprocess_image(sam_img).unsqueeze(0)

        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).long().unsqueeze(0).unsqueeze(0)

        question = random.choice(self.question_list).format(class_name=obj_name.lower())
        answer = random.choice(self.answer_list).format(body_parts=body_parts)
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()

        return (
            img_path,       # path
            sam_img,        # (1, 3, H, W)
            clip_img,       # CLIP image
            [conversation], # list of convs
            mask,           # binary mask (1, 1, H, W)
            mask[0, 0],     # (H, W)
            torch.zeros(0), # dummy gt_contact_3d
            torch.zeros(1, 5), # dummy cam_params
            resize,         # resize
            [question],
            [obj_name],
            'h2dcontact',
            [mask_path],
        )
