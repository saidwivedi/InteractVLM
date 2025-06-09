import os
import random
from os.path import isfile
import sys
import cv2
import tqdm
import numpy as np
import torch
import joblib as jl
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor

# Dynamically add the necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
model_path = os.path.join(project_root, 'model')
utils_path = os.path.join(project_root, 'utils')
data_path = os.path.join(project_root, 'preprocess_data')

sys.path.append(project_root)
sys.path.append(model_path)
sys.path.append(utils_path)
sys.path.append(data_path)

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from preprocess_data.constants import OBJS_VIEW_DICT, HUMAN_VIEW_DICT

from utils.utils import SAM_MEAN_PIXEL, SAM_STD_PIXEL, IGNORE_LABEL

from os.path import join, isdir, isfile, basename


def normalize_cam_params(cam_params):
    if cam_params is None:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    distance, elevation, azimuth, x_translation, y_translation = cam_params
    
    # Normalize each parameter
    normalized_distance = distance / 10.0  # Assuming max distance is 10
    normalized_elevation = elevation / 360.0  # Elevation in degrees
    normalized_azimuth = azimuth / 360.0  # Azimuth in degrees
    normalized_x_translation = (x_translation + 1.0) / 2.0  # Assuming x_translation is between -1 and 1
    normalized_y_translation = (y_translation + 1.0) / 2.0  # Assuming y_translation is between -1 and 1
    
    return torch.tensor([normalized_distance, normalized_elevation, normalized_azimuth, normalized_x_translation, normalized_y_translation])

class BaseContactSegDataset(torch.utils.data.Dataset):
    
    pixel_mean = torch.Tensor(SAM_MEAN_PIXEL).view(-1, 1, 1)
    pixel_std = torch.Tensor(SAM_STD_PIXEL).view(-1, 1, 1)
    ignore_label = IGNORE_LABEL

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        image_size=1024,
    ):
        self.base_image_dir = "./data/"
        self.ignore_label = IGNORE_LABEL
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.obj_view_dict_all = OBJS_VIEW_DICT
        self.human_view_dict_all = HUMAN_VIEW_DICT

        self.llava_augment = transforms.Compose([
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0), 
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            transforms.RandomAutocontrast(p=0.2)
        ])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def concatenate_images_grid(self, image_paths, grid_size=(1, 2, 2), mask_size=(512, 512)):
        """Concatenate images into a grid or stack them in channels."""
        images = [Image.open(image) for image in image_paths]

        if images[0].size != mask_size:
            print(f'Image size mismatch: {images[0].size} vs {mask_size}')
            images = [image.resize(mask_size, Image.NEAREST) for image in images]
        
        if grid_size[0] > 1:
            concat_images = np.stack([np.asarray(image) for image in images], axis=0)
            return concat_images[..., np.newaxis] if concat_images.ndim == 3 else concat_images
        else:
            # Calculate width and height of the 2D grid
            grid_size = (grid_size[1], grid_size[2])
            img_width, img_height = images[0].size
            grid_width = grid_size[0] * img_width
            grid_height = grid_size[1] * img_height

            # Create a new blank image with the right size
            grid_image = Image.new('RGB', (grid_width, grid_height))

            # Paste each image into the grid
            for index, image in enumerate(images):
                row = index // grid_size[0]
                col = index % grid_size[0]
                grid_image.paste(image, (col * img_width, row * img_height))

            return np.array(grid_image)

    def load_and_prepare_label(self, label_paths, valid_regions=None, grid_size=(1, 2, 2), mask_size=(512, 512)):
        """Load and prepare label."""

        label = self.concatenate_images_grid(label_paths, grid_size=grid_size, mask_size=mask_size)
        label = label.astype(np.int32)
        if label.ndim == 3:
            label = label[np.newaxis, ...] # Add view dimension (V, H, W, 1)

        label = label[..., 0]  # Take only one channel

        # Setting outside valid regions to -1
        if valid_regions is not None:
            label[valid_regions == 0] = self.ignore_label

        # Make sure the label is binary
        label[label == 255] = 1

        # Check the proportion of meaningful pixels
        num_contact_pixels = np.sum(label == 1)
        total_pixels = label.size
        contact_proportion = num_contact_pixels / total_pixels

        return label, contact_proportion
    
    def load_and_prepare_heatmap(self, label_paths, valid_regions=None, grid_size=(1, 2, 2), mask_size=(1024, 1024)):
        """Load and prepare heatmap."""

        label = self.concatenate_images_grid(label_paths, grid_size=grid_size, mask_size=mask_size)
        label = (label/255.).astype(np.float32)
        if label.ndim == 3:
            label = label[np.newaxis, ...] # Add view dimension (V, H, W, 1)

        label = label[..., 0]  # Take only one channel

        # Setting outside valid regions to -1
        if valid_regions is not None:
            label[valid_regions == 0] = self.ignore_label

        return label, None


    def load_and_prepare_sam_image(self, sam_image_paths, grid_size=(1, 2, 2), mask_size=(512, 512)):
        """Prepare SAM images."""
        sam_images = self.concatenate_images_grid(sam_image_paths, grid_size=grid_size, mask_size=mask_size)

        valid_regions = []      # (V, 1, H, W)
        for sam_image in sam_images:
            valid_regions.append((sam_image.sum(axis=-1) < 255 * 3).astype(np.uint8))
        valid_regions = np.stack(valid_regions) 

        if grid_size[0] > 1:
            sam_images = [self.transform.apply_image(image) for image in sam_images]
            sam_images = torch.stack([self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()) for image in sam_images])
        else:
            sam_images = self.transform.apply_image(sam_images)
            sam_images = self.preprocess(torch.from_numpy(sam_images).permute(2, 0, 1).contiguous())
        
        resize = sam_images.shape[2:]
        return sam_images, valid_regions, resize
    
    def load_and_prepare_clip_image(self, image_path):
        """Load and prepare CLIP image."""
        llava_image = cv2.imread(image_path)    
        llava_image = cv2.cvtColor(llava_image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(llava_image, return_tensors="pt")["pixel_values"][0]
        # if self.is_train:
        #     image_clip = self.llava_augment(image_clip)
        return image_clip   

    def generate_h_conversations(self, sampled_classes, body_parts=None):
        """Generate conversation prompts and answers based on classes."""
        questions = []
        answers = []
        for sampled_cls in sampled_classes:

            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=sampled_cls.lower()))

            answers.append(random.choice(self.answer_list).format(body_parts=body_parts))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        for i in range(len(questions)):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())

        return conversations, questions
    
    def generate_o_conversations(self, sampled_classes, affordance):
        """Generate conversation prompts and answers based on classes."""
        questions = []
        answers = []
        for sampled_cls in sampled_classes:

            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=sampled_cls.lower()))
            
            answers.append(random.choice(self.answer_list).format(affordance=affordance, 
                                                                  class_name=sampled_cls.lower()))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        for i in range(len(questions)):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())

        return conversations, questions

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this!")