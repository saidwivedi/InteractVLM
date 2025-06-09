import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor
import torchshow as ts


from .hcontact_3d import HContactSegDataset
from .ocontact_3d import OContactSegDataset, OAffordSegDataset
from .hcontact_2d import H2DContactSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset
from .dataset import ValDataset
from utils.utils import IGNORE_LABEL

def print_batch(batch):
    print(f'\nLlava image path: {batch[0]}')
    print(f'SAM mask path: {batch[12]}')
    print(f'Shape of sam_image: {batch[1].shape}, Shape of llava_image: {batch[2].shape}')
    print(f'Shape of sam_mask: {batch[4].shape}')
    print(f'gt_contact_3d: {batch[6].shape} with {len(batch[6].nonzero())} non-zero elements')
    print(f'cam_params: {batch[7].shape}')
    print(f'Prompt: {batch[3]}')
    print(f'Unique values in sam_mask: {np.unique(batch[4])}')

def test_dataloader_loading():

    mock_tokenizer = lambda x: x
    mock_vision_tower = 'openai/clip-vit-large-patch14'
    max_num_images = 10

    contact_dataset_config = {
        "oC_sam_view_type": '4MV-Z_HM_BM', # '4MV-XY_Rand'
        "oC_sam_input_type": 'color',
        "oC_ranking": 'lookup',  
        "oC_question_type": 'afford_obj',
        "hC_sam_view_type": '4MV-Z_Vitru', # '4MV-XY_Fix'
        "hC_sam_input_type": 'norm',
        "hC_mask_type": 'objectwise',
        "hC_question_type": 'parts',
        "hC_train_fraction": 0.5,
        "token_type": 'Gen-DifDe',
    }

    oafford_seg_data = 'piad_oafford'
    ocontact_seg_data = 'pico_ocontact'
    hcontact_seg_data = 'damon_hcontact'
    vqa_data = 'damon_hcontact'

    display_datasets = ['human_val']

    if 'human2d_train' in display_datasets:
        contact_train_dataset = H2DContactSegDataset(
            base_image_dir='path/to/base_image_dir',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            is_train=True,
            image_size=1024,
            question_type='simple',
        )

        # Get an item from the dataset to ensure it loads correctly
        for i in range(max_num_images):
            disp_imgs = []
            sample = contact_train_dataset[i]
            sample[4][sample[4] == IGNORE_LABEL] = 0
            print_batch(sample)
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            disp_imgs.extend([sample[2]]) 
            disp_imgs.extend([sample[4]] if sample[4].ndim < 4 else [x for x in sample[4]])
            ts.show(disp_imgs)


    # ################ OAffordSeg -- train ################

    if 'obj_train' in display_datasets:
        contact_train_dataset = OAffordSegDataset(
            base_image_dir='path/to/base_image_dir',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            contact_dataset_config=contact_dataset_config,
            is_train=True,
            samples_per_epoch=10,
            image_size=1024,
            num_classes_per_sample=3,
            contact_seg_data=oafford_seg_data
        )

        # Get an item from the dataset to ensure it loads correctly
        for i in range(max_num_images):
            disp_imgs = []
            sample = contact_train_dataset[i]
            sample[4][sample[4] == IGNORE_LABEL] = 0
            print_batch(sample)
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            disp_imgs.extend([sample[2]]) 
            disp_imgs.extend([sample[4]] if sample[4].ndim < 4 else [x for x in sample[4]])
            ts.show(disp_imgs)

    # ############### OContactSeg -- train ################

    if 'obj_train' in display_datasets:
        contact_train_dataset = OContactSegDataset(
            base_image_dir='path/to/base_image_dir',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            contact_dataset_config=contact_dataset_config,
            is_train=True,
            samples_per_epoch=10,
            image_size=1024,
            num_classes_per_sample=3,
            contact_seg_data=ocontact_seg_data,
            val_dataset=ocontact_seg_data,
        )

        # Get an item from the dataset to ensure it loads correctly
        for i in range(max_num_images):
            disp_imgs = []
            sample = contact_train_dataset[i]
            sample[4][sample[4] == IGNORE_LABEL] = 0
            print_batch(sample)
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            disp_imgs.extend([sample[2]]) 
            disp_imgs.extend([sample[4]] if sample[4].ndim < 4 else [x for x in sample[4]])
            ts.show(disp_imgs)

    
    # # # ############### OAffordSeg -- val ################

    if 'obj_val' in display_datasets:
        contact_val_dataset = OAffordSegDataset(
            base_image_dir='path/to/base_image_dir',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            contact_dataset_config=contact_dataset_config,
            is_train=False,
            val_dataset=oafford_seg_data,
            image_size=1024,
        )

        print(f'lenth of val_dataset: {len(contact_val_dataset)}')
        for i in range(max_num_images):
            disp_imgs = []
            sample = contact_val_dataset[i]
            print_batch(sample)
            sample[4][sample[4] == IGNORE_LABEL] = 0
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            disp_imgs.extend([sample[2]])
            disp_imgs.extend([sample[4]] if sample[4].ndim < 4 else [x for x in sample[4]])
            ts.show(disp_imgs)

    # ############### HContactSeg -- train ################

    if 'human_train' in display_datasets:
        contact_train_dataset = HContactSegDataset(
            base_image_dir='path/to/base_image_dir',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            contact_dataset_config=contact_dataset_config,
            is_train=True,
            image_size=1024,
            samples_per_epoch=10,
            num_classes_per_sample=3,
            contact_seg_data=hcontact_seg_data,
        )

        # # Get an item from the dataset to ensure it loads correctly
        for i in range(max_num_images):
            disp_imgs = []
            sample = contact_train_dataset[i]
            sample[4][sample[4] == IGNORE_LABEL] = 0
            print_batch(sample)
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            disp_imgs.extend([sample[2]]) 
            disp_imgs.extend([sample[4]] if sample[4].ndim < 4 else [x for x in sample[4]])
            ts.show(disp_imgs)


    ############### HContactSeg -- val ################

    if 'human_val' in display_datasets:
        contact_val_dataset = HContactSegDataset(
            base_image_dir='path/to/base_image_dir',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            contact_dataset_config=contact_dataset_config,
            is_train=False,
            image_size=1024,
            val_dataset=hcontact_seg_data,
        )

        print(f'lenth of val_dataset: {len(contact_val_dataset)}')

        # # Get an item from the dataset to ensure it loads correctly
        for i in range(max_num_images):
            disp_imgs = []
            sample = contact_val_dataset[i]
            sample[4][sample[4] == IGNORE_LABEL] = 0
            print_batch(sample)
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            disp_imgs.extend([sample[2]]) 
            disp_imgs.extend([sample[4]] if sample[4].ndim < 4 else [x for x in sample[4]])
            ts.show(disp_imgs)

    # ################ ReasonSeg ################

    if 'reason_val' in display_datasets:
        val_dataset = ValDataset(
            base_image_dir='./dataset',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            val_dataset="ReasonSeg|val",
            image_size=1024,
        )

        sample = val_dataset[0]
        print_batch(sample)
        ts.show([sample[1], sample[2], sample[4]])

    # ################ VQA Dataset ################

    if 'vqa' in display_datasets:    
        vqa_dataset = VQADataset(
            base_image_dir='./dataset',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            samples_per_epoch=10,
            image_size=1024,
            num_classes_per_sample=3,
            exclude_val=False,
            vqa_data=vqa_data,
        )
        for i in range(max_num_images):
            disp_imgs = []
            sample = vqa_dataset[i]
            print_batch(sample)
            disp_imgs.extend([sample[1]] if sample[1].ndim < 4 else [x for x in sample[1]])
            ts.show(disp_imgs)


    # ################ ADE20K ################

    if 'sem_seg' in display_datasets:
        sem_dataset = SemSegDataset(
            base_image_dir='./dataset',
            tokenizer=mock_tokenizer,
            vision_tower=mock_vision_tower,
            samples_per_epoch=10,
            image_size=1024,
            num_classes_per_sample=4,
            sem_seg_data="ade20k",
        )

        sample = sem_dataset[0]
        print_batch(sample)
        ts.show([sample[1], sample[2], sample[4][0, :, :], sample[5][0, :, :]])


if __name__ == '__main__':
    test_dataloader_loading()
