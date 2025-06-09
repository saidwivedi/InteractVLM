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
from os.path import join, isdir, isfile, basename

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

from utils.utils import HCONTACT_ANSWER_LIST
from utils.utils import HCONTACT_QUESTION_LIST
from utils.utils import HCONTACT_PARTS_QUESTION_LIST, HCONTACT_PARTS_ANSWER_LIST
from utils.utils import SAM_MEAN_PIXEL, SAM_STD_PIXEL, IGNORE_LABEL

from .base_contact_dataset import BaseContactSegDataset, normalize_cam_params

def check_paths_exist(paths, printstr=''):
    for path in paths:
        if not isfile(path):
            print(f'File does not exist: {path} - {printstr}')
            return False
    return True

def init_damon_hcontact(base_image_dir, view_dict, split='Train', sam_input_type='grey', contact_mask_type='objectwise', train_fraction=1.0):   

    folderpath, grid_size   = view_dict['folder'], view_dict['grid_size']
    num_vertices = view_dict['num_vertices']
    cam_params_dict = view_dict['cam_params']
    view_names = view_dict['names'].flatten()

    base_image_dir = join(base_image_dir, f'damon/{split}')
    img_list = np.load(join(base_image_dir, f'imgname.npy'), allow_pickle=True)
    llava_images = [join(base_image_dir, f'images/{os.path.basename(f)}') for f in img_list]

    classes = []
    gt_contact_3d, labels, valid_llava_images, body_parts = [], [], [], []
    num_examples_per_class = {}

    if contact_mask_type == 'all_contact': ### TODO: I think it is deprecated, check if the model is trained in this mode 

        all_contact_annot = np.load(join(base_image_dir, f'contact_label.npy'), allow_pickle=True)
        body_parts_annot = jl.load(join(base_image_dir, f'body_parts_all.pkl'))
        for idx, llava_image in tqdm.tqdm(enumerate(llava_images)):
            
            base_name = os.path.basename(llava_image)[:-4]
            # process all contact masks
            mask_all_contact = [
                join(base_image_dir, folderpath, 'all_contact', f'{base_name}_{view}.png')
                for view in view_names
            ]
            if not check_paths_exist(mask_all_contact, printstr=f'all_contact_{idx}'):
                continue
            
            vertices_all_contact = torch.tensor(all_contact_annot[idx]).int()

            body_parts_sample = ', '.join(body_parts_annot[base_name])
            body_parts.append(body_parts_sample)
            
            labels.append(mask_all_contact)
            gt_contact_3d.append(vertices_all_contact)
            valid_llava_images.append(llava_image)

        classes = [['all the objects or ground']] * len(valid_llava_images) # objname consistent with questions

    elif contact_mask_type == 'objectwise':

        objectwise_contact_annot = np.load(join(base_image_dir, f'contact_label_objectwise.npy'), allow_pickle=True)
        body_parts_annot = jl.load(join(base_image_dir, f'body_parts_objectwise.pkl'))
        for idx, llava_image in tqdm.tqdm(enumerate(llava_images)):

            base_name = os.path.basename(llava_image)[:-4]
            
            # Process object-wise contacts
            for obj_name, contact_vertices in objectwise_contact_annot[idx].items():
                
                contact_array = torch.zeros(num_vertices).int()
                if len(contact_vertices) == 0:
                    continue
                contact_array[contact_vertices] = 1
                    
                obj_masks = [
                    join(base_image_dir, folderpath, obj_name, f'{base_name}_{view}.png')
                    for view in view_names
                ]
                if not check_paths_exist(obj_masks, printstr=f'objectwise_{obj_name}_{idx}'):
                    continue
                # Count number of examples of contact per object category
                if obj_name not in num_examples_per_class.keys():
                    num_examples_per_class[obj_name] = 0
                num_examples_per_class[obj_name] += 1

                # Get body parts name in contact
                body_parts_sample = ', '.join(body_parts_annot[f'{base_name}_{obj_name}'])
                body_parts.append(body_parts_sample)
                
                # Replace certain keywords in the object name
                ##### TODO: Check if this is necessary #####
                if 'supporting' in obj_name:
                    # continue
                    obj_name = obj_name.replace('supporting', 'support object or ground')

                labels.append(obj_masks)
                gt_contact_3d.append(contact_array)
                valid_llava_images.append(llava_image)
                classes.append([obj_name])

    # Filter data for training split
    if split.lower() == 'train' and train_fraction < 1.0:
        total_samples = len(valid_llava_images)
        num_train_samples = int(total_samples * train_fraction)
        # Use random seed for reproducibility
        np.random.seed(42)
        selected_indices = np.random.choice(total_samples, num_train_samples, replace=False)
        selected_indices.sort()  # Sort indices for consistency
        
        valid_llava_images = [valid_llava_images[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        gt_contact_3d = [gt_contact_3d[i] for i in selected_indices]
        classes = [classes[i] for i in selected_indices]
        body_parts = [body_parts[i] for i in selected_indices]
        
        print(f"Selected {len(valid_llava_images)} samples for training out of {total_samples} total samples")
        
        # Recalculate num_examples_per_class for the selected samples
        num_examples_per_class = {}
        for cls in classes:
            obj_name = cls[0]
            if obj_name not in num_examples_per_class:
                num_examples_per_class[obj_name] = 0
            num_examples_per_class[obj_name] += 1
                
    total_examples = sum(num_examples_per_class.values())
    print(f'Number of examples per class in {split} for HContact DAMON with {len(list(num_examples_per_class.keys()))} classes')
    print(f'{num_examples_per_class} with total examples: \nTotal number of samples: {total_examples}')

    sam_images_path, cam_params_list = [], []
    for view_name in view_names:
        sam_images_path.append(join(base_image_dir, f'{folderpath}/body_render_{sam_input_type}_{view_name}.png'))
        cam_params_list.append(normalize_cam_params(cam_params_dict[view_name]))

    cam_params = [cam_params_list] * len(valid_llava_images)

    return classes, valid_llava_images, labels, sam_images_path, gt_contact_3d, cam_params, body_parts
    

def init_lemon_hcontact(base_image_dir, view_dict, split='train', sam_input_type='grey'):
    
    folderpath, grid_size   = view_dict['folder'], view_dict['grid_size']
    num_vertices = view_dict['num_vertices']
    cam_params_dict = view_dict['cam_params']
    view_names = view_dict['names'].flatten()

    classes, labels, valid_llava_images, body_parts = [], [], [], []
    img_list = open(join(base_image_dir, f'lemon/txt_scripts/{split}.txt')).read().splitlines()
    body_parts_annot = jl.load(join(base_image_dir, f'lemon/body_parts_{split}.pkl'))

    llava_images = [join(base_image_dir, f) for f in img_list]

    gt_contact_3d = []
    num_examples_per_class = {}
    num_zero_contact = 0
    for idx, llava_image in tqdm.tqdm(enumerate(llava_images)):
        mask_list = []

        # Count number of examples per class
        object_name = basename(llava_image).split('_')[0]
        if object_name not in num_examples_per_class.keys():
            num_examples_per_class[object_name] = 0
        num_examples_per_class[object_name] += 1

        contact_vertices = jl.load(llava_image.replace('Images', 'smplh_contact_pkl')[:-4] + '.pkl')
        if contact_vertices.nonzero()[0].shape[0] == 0:
            print(f'Warning: No contact pixels in the mask for {llava_image}')
            continue

        # Generate mask paths for each view
        for view_name in view_names:
            mask_list.append(llava_image.replace('Images', folderpath)[:-4] + f'_{view_name}.png')

        body_parts_sample = ', '.join(body_parts_annot[basename(llava_image)[:-4]])
        body_parts.append(body_parts_sample)    

        labels.append(mask_list)
        valid_llava_images.append(llava_image)
        gt_contact_3d.append(torch.from_numpy(contact_vertices).int())
        classes.append([object_name])
    
    print(f'Number of images with zero contact: {num_zero_contact}')

    total_examples = sum(num_examples_per_class.values())
    print(f'Number of examples per class in {split} for HContact Lemon: {num_examples_per_class} with total examples: {total_examples}')

    sam_images_path, cam_params_list = [], []
    for view_name in view_names:
        sam_images_path.append(join(base_image_dir, f'lemon/{folderpath}/body_render_{sam_input_type}_{view_name}.png'))
        cam_params_list.append(normalize_cam_params(cam_params_dict[view_name]))

    cam_params = [cam_params_list] * len(valid_llava_images)

    return classes, valid_llava_images, labels, sam_images_path, gt_contact_3d, cam_params, body_parts


class HContactSegDataset(BaseContactSegDataset):

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        contact_dataset_config,
        is_train=True,
        image_size=224,
        samples_per_epoch=500 * 8 * 2 * 10,
        num_classes_per_sample=3,
        contact_seg_data="lemon_hcontact",
        val_dataset=None
    ):
        super().__init__(base_image_dir, tokenizer, vision_tower, image_size)
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch if is_train else None
        self.num_classes_per_sample = num_classes_per_sample

        self.contact_seg_data = sorted(contact_seg_data.split("||") if is_train else val_dataset.split("||"))
        print(f'-----> Hcontact_seg_data in {"training" if is_train else "validation"}: {self.contact_seg_data}')

        self.sam_input_type = contact_dataset_config['hC_sam_input_type']
        self.sam_view_type = contact_dataset_config['hC_sam_view_type'] 
        self.contact_mask_type = contact_dataset_config['hC_mask_type']
        self.train_fraction = contact_dataset_config['hC_train_fraction'] if 'hC_train_fraction' in contact_dataset_config else 1.0
        self.view_dict = self.human_view_dict_all[self.sam_view_type]
        self.grid_size = self.view_dict['grid_size']
        print(f'----> Grid_size for SAM images: {self.grid_size}')
        self.mask_size = (self.view_dict['mask_size'], self.view_dict['mask_size'])
        self.question_type = contact_dataset_config['hC_question_type']

        if self.question_type == 'simple':
            self.answer_list = HCONTACT_ANSWER_LIST
            self.short_question_list = HCONTACT_QUESTION_LIST
        elif self.question_type == 'parts':
            self.answer_list = HCONTACT_PARTS_ANSWER_LIST
            self.short_question_list = HCONTACT_PARTS_QUESTION_LIST
        
        base_token_type = contact_dataset_config['token_type'].replace('-DifDe', '')
        if base_token_type == 'Gen-Int':
            self.answer_list = [ans.replace('HTOKEN', 'ISEG') for ans in self.answer_list]
        elif base_token_type == 'Gen-Hu-Obj':
            self.answer_list = [ans.replace('HTOKEN', 'HSEG') for ans in self.answer_list]
        else:
            self.answer_list = [ans.replace('HTOKEN', 'SEG') for ans in self.answer_list]
            
        self.sam_dict, self.data2list, self.data2classes, self.ds_size = {}, {}, {}, {}

        for ds in self.contact_seg_data:
            # Initialize LEMON Human Contact dataset
            if 'lemon' in ds:
                split = 'train' if is_train else 'val'
                print(f'\nInitializing dataset: {ds} with split: {split}')
                classes, llava_images, labels, sam_images_path, gt_contact_3d, cam_params, body_parts = \
                    init_lemon_hcontact(self.base_image_dir, self.view_dict, split=split, sam_input_type=self.sam_input_type)
                print(f'---> Initialized dataset: {ds} with {len(llava_images)} images')
            # Initialize DAMON Human Contact dataset
            elif 'damon' in ds:
                split = 'train' if is_train else 'test'
                print(f'\nInitializing dataset: {ds} with split: {split}')
                classes, llava_images, labels, sam_images_path, gt_contact_3d, cam_params, body_parts = \
                    init_damon_hcontact(self.base_image_dir, self.view_dict, split, self.sam_input_type, self.contact_mask_type, self.train_fraction)
                print(f'---> Initialized dataset: {ds} with {len(llava_images)} images')
            # Since we are using fixed SAM images, we can load them once and use them for all samples
            self.sam_images, self.valid_regions, self.resize = \
                self.load_and_prepare_sam_image(sam_images_path, self.grid_size, self.mask_size)
            self.sam_dict[ds] = (self.sam_images, self.valid_regions, self.resize)
            self.data2list[ds] = (llava_images, labels, gt_contact_3d, cam_params, body_parts)
            self.data2classes[ds] = classes
            self.ds_size[ds] = len(llava_images)
            assert len(llava_images) == len(labels) == len(gt_contact_3d) == len(cam_params) == len(classes) == len(body_parts), \
                  f"Mismatch in dataset {ds} llava_images: {len(llava_images)}, labels: {len(labels)}, gt_contact_3d: {len(gt_contact_3d)}, cam_params: {len(cam_params)}, classes: {len(classes)}, body_parts: {len(body_parts)}"

    def __len__(self):
        if self.is_train:
            return self.samples_per_epoch
        else:
            return sum([self.ds_size[ds] for ds in self.contact_seg_data])

    def _get_train_sample(self, idx):
        ds = random.choice(self.contact_seg_data)
        llava_images, label_list, gt_contact_3d_all, cam_params_all, body_parts_all = self.data2list[ds]
        classes_list = self.data2classes[ds]
        idx = random.randint(0, len(llava_images) - 1)
        llava_image = llava_images[idx]
        label_paths = label_list[idx]
        gt_contact_3d = gt_contact_3d_all[idx]
        cam_params = cam_params_all[idx]
        body_parts = body_parts_all[idx]
        sampled_classes = classes_list[idx]
        return llava_image, label_paths, gt_contact_3d, cam_params, body_parts, sampled_classes, ds
    
    def _get_val_sample(self, idx):
        total_images = 0
        for ds in self.contact_seg_data:
            ds_size = self.ds_size[ds]
            if idx < total_images + ds_size:
                llava_images, label_list, gt_contact_3d_all, cam_params_all, body_parts_all = self.data2list[ds]
                classes_list = self.data2classes[ds]
                idx = idx - total_images
                break
            total_images += ds_size
        llava_image = llava_images[idx]
        label_paths = label_list[idx]
        gt_contact_3d = gt_contact_3d_all[idx]
        cam_params = cam_params_all[idx]
        body_parts = body_parts_all[idx]
        sampled_classes = classes_list[idx]

        return llava_image, label_paths, gt_contact_3d, cam_params, body_parts, sampled_classes, ds

    def __getitem__(self, idx):
        
        llava_image_path, label_paths, gt_contact_3d, cam_params, body_parts, sampled_classes, ds = \
            self._get_train_sample(idx) if self.is_train else self._get_val_sample(idx)
        
        # SAM images, valid regions and resize are fixed for each dataset
        self.sam_images, self.valid_regions, self.resize = self.sam_dict[ds]

        # Load and prepare valid masks
        label, contact_proportion = self.load_and_prepare_label(label_paths, self.valid_regions, self.grid_size, self.mask_size)

        # Load and prepare CLIP image
        image_clip = self.load_and_prepare_clip_image(llava_image_path)

        unique_label = np.unique(label).tolist()
        if self.ignore_label in unique_label:
            unique_label.remove(self.ignore_label)
        
        if self.is_train and len(unique_label) == 0:
            print(f'Warning: No contact pixels in the mask for {llava_image_path}')
            return self.__getitem__(random.randint(0, self.samples_per_epoch - 1))

        conversations, questions = self.generate_h_conversations(sampled_classes, body_parts)

        cam_params = torch.stack(cam_params).float() 
        label = torch.from_numpy(label)
        masks = label.unsqueeze(1)
        inference = False if self.is_train else True

        results = (
            llava_image_path,
            self.sam_images,    # (V, C, H, W)
            image_clip,
            conversations,
            masks,              # (V, 1, H, W)
            label[0],           # (H, W) used to get original mask
            gt_contact_3d,
            cam_params,         # (V, 5) normalized camera parameters
            self.resize,
            questions if self.is_train else None,
            sampled_classes,
            ds,
            label_paths,
        )
        if not self.is_train:
            results = results + (inference,)

        return results


