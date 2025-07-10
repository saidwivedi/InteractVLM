
import os
import random
from os.path import isfile
import sys
import numpy as np
import torch
import joblib as jl
from os.path import join, isfile

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

from preprocess_data.constants import VALID_OBJ_NAMES_LEMON, AFFORD_PROB_LEMON, VALID_OBJ_NAMES_PIAD, AFFORD_PROB_PIAD, AFFORD_LIST_PIAD, AFFORD_LIST_LEMON
from preprocess_data.generate_lemon_obj_heatmap import extract_point_file_lemon
from preprocess_data.generate_piad_obj_heatmap import extract_point_file_piad

from utils.utils import OAFFORD_ANSWER_LIST, OCONTACT_ANSWER_LIST
from utils.utils import OAFFORD_QUESTION_LIST, OCONTACT_QUESTION_LIST
from utils.utils import OAFFORD_AFFORD_QUESTION_LIST, OAFFORD_AFFORD_ANSWER_LIST, OAFFORD_AFFORD_OBJ_ANSWER_LIST

from .base_contact_dataset import BaseContactSegDataset, normalize_cam_params

def get_objname_afford(llava_image_path):
    if 'piad' in llava_image_path:
        sample_name = llava_image_path.split('/')[-1].split('_')[2:]
        base_rendObj_path = os.path.dirname(llava_image_path).replace('Img', f'insert_path')
        obj_name = sample_name[0]
        afford_name = sample_name[1]
    elif 'pico' in llava_image_path:
        sample_name = llava_image_path.split('/')[-1].split('__')
        base_rendObj_path = os.path.dirname(llava_image_path).replace('images', f'insert_path')
        obj_name = sample_name[0]
        afford_name = None
    elif 'lemon' in llava_image_path:
        sample_name = llava_image_path.split('/')[-1].split('_')
        base_rendObj_path = os.path.dirname(llava_image_path).replace('Images', f'lemon_ocontact/insert_path')
        obj_name = sample_name[0]
        afford_name = sample_name[1]

    return obj_name, afford_name, base_rendObj_path

def filter_oafford_images(img_list, obj_names, afford_dict, ignore_keywords):
    valid_images = []
    classes = []
    valid_obj_names = [obj_name for obj_name in obj_names if obj_name not in ignore_keywords]
    num_examples_per_class = {obj : 0 for obj in valid_obj_names}
    valid_affordances = set()
    for path in img_list:
        obj_name, _, _ = get_objname_afford(path)
        path_lower = path.lower()
        # Find the object in the path
        obj_in_path = next((obj for obj in valid_obj_names if obj.lower() in path_lower), None)
        if obj_in_path:
            # If object is found, check if the affordance in the path is valid for this object
            valid_afford = afford_dict[obj_in_path].keys()
            valid_afford = [afford for afford in valid_afford if afford not in ignore_keywords]
            if any(afford.lower() in path_lower for afford in valid_afford):
                valid_images.append(path)
                num_examples_per_class[obj_name] += 1
                classes.append([obj_name])
                valid_affordances.update(valid_afford)
    # num_examples_per_class = {k: np.round(v/len(img_list), 4) for k, v in num_examples_per_class.items()}
    return valid_images, classes, valid_obj_names, valid_affordances, num_examples_per_class


def init_ocontact(base_image_dir, dataset, split='Train', ignore_keywords=[]): 

    # Object Contact with Mesh -- PICO
    if 'pico_ocontact' in dataset:
        split = 'train' if split == 'Train' else 'test'
        img_list = jl.load(join(base_image_dir, f'pico/{split}/{split}_imglist.pkl'))

        object_match, classes = {}, []
        llava_images = img_list
        object_match = {f: f"{os.path.basename(f).split('__')[-1][:-4]}" for f in img_list}
        classes = [[f"{os.path.basename(f).split('__')[0]}"] for f in img_list]

        return classes, llava_images, object_match
        
    ## Object Affordance with Point Clouds -- PIAD, LEMON
    if 'piad_oafford' in dataset:
        img_list = open(join(base_image_dir, f'piad_ocontact_seen/Img_{split}.txt')).read().splitlines()
        lookup_meshes_f = join(base_image_dir, f'piad_ocontact_seen/piad_lookup_results_{split}.pkl')
        test_obj_f = join(base_image_dir, f'piad_ocontact_seen/Point_{split}.txt')
        obj_names, afford_dict = VALID_OBJ_NAMES_PIAD, AFFORD_PROB_PIAD
    elif 'piad_unseen_oafford' in dataset:
        img_list = open(join(base_image_dir, f'piad_ocontact_unseen/Img_{split}.txt')).read().splitlines()
        lookup_meshes_f = join(base_image_dir, f'piad_ocontact_unseen/piad_unseen_lookup_results_{split}.pkl')
        test_obj_f = join(base_image_dir, f'piad_ocontact_unseen/Point_{split}.txt')
        obj_names, afford_dict = VALID_OBJ_NAMES_PIAD, AFFORD_PROB_PIAD 
    elif 'lemon_oafford' in dataset:
        split = 'train' if split == 'Train' else 'val'
        img_list = open(join(base_image_dir, f'lemon/txt_scripts/{split}.txt')).read().splitlines()
        lookup_meshes_f = join(base_image_dir, f'lemon/lemon_ocontact/lemon_lookup_results_{split}.pkl')
        test_obj_f = join(base_image_dir, f'lemon/txt_scripts/Point_{split}.txt')
        obj_names, afford_dict = VALID_OBJ_NAMES_LEMON, AFFORD_PROB_LEMON

    ### For training, there is no 1:1 mapping, hence we use Openshape lookup
    if split == 'Train' or split == 'train':
        print(f'total images before filtering: {len(img_list)}')
        llava_images, classes, valid_obj_names, valid_affordances, num_examples_per_class = \
            filter_oafford_images(img_list, obj_names, afford_dict, ignore_keywords)
        print(f'total images after filtering: {len(llava_images)}')
        print(f'Number of examples per class in {split} for OAfford {dataset}: {num_examples_per_class}')

        print(f'valid_obj_names: {valid_obj_names}')
        print(f'valid_affordances: {valid_affordances}')
        print(f'lookup_meshes_f: {lookup_meshes_f} {isfile(lookup_meshes_f)}')
        # object_match contains matches for all images irrespective of if they are valid or not
        # since keys are filenames, indexing won't be a problem
        object_match = jl.load(lookup_meshes_f)

    else:
        object_match, classes = {}, []
        llava_images = img_list
        obj_files = open(test_obj_f, 'r').read().splitlines()
        for idx, img_f in enumerate(llava_images):
            obj_name, _, _ = get_objname_afford(img_f)
            classes.append([obj_name])
            object_match[img_f] = [obj_files[idx]]

    return classes, llava_images, object_match

def get_sam_input_and_label_ocontact(llava_image_path, object_match, view_dict):

    folderpath, view_names  = view_dict['mesh_folder'], view_dict['names']
    cam_params_dict = view_dict['mesh_cam_params']
    view_names = np.array(view_names)
    obj_name, afford_name, base_rendObj_path = get_objname_afford(llava_image_path)
    base_rendObj_path = base_rendObj_path.replace('insert_path', f'{folderpath}/{obj_name}')
    sample_name = object_match[llava_image_path]

    gt_contact_3d = jl.load(join(base_rendObj_path, f'contact_vertices_{sample_name}.pkl'))
        
    sel_view_names = view_names.flatten()  # Default value
    sam_images, label_paths, cam_params_list = [], [], []
    all_files_exist = True  # Flag to track if all required files exist

    for view_name in sel_view_names:
        sam_image = os.path.join(base_rendObj_path, f'sam_{sample_name}_{view_name}.png')
        label_path = os.path.join(base_rendObj_path, f'mask_{sample_name}_{view_name}.png')
        sam_images.append(sam_image)
        label_paths.append(label_path)
        cam_params_list.append(normalize_cam_params(cam_params_dict[view_name]))
        if not os.path.isfile(sam_image) or not os.path.isfile(label_path):
            print(f'File does not exist: {sam_image} or {label_path}')
            all_files_exist = False
            break

    if all_files_exist:
        return sam_images, label_paths, cam_params_list, gt_contact_3d, afford_name

    print(f'\nNo valid data found for {os.path.basename(llava_image_path)}')
    return None, None, None, None, None

def get_sam_input_and_label_oafford(is_train, llava_image_path, object_match, object_ranking, view_dict, sam_input_type='depth'):

    folderpath, grid_size, order, view_names  = view_dict['folder'], view_dict['grid_size'], view_dict['order'], view_dict['names']
    cam_params_dict = view_dict['cam_params']
    view_names = np.array(view_names)
    obj_name, afford_name, base_rendObj_path = get_objname_afford(llava_image_path)
    base_rendObj_path = base_rendObj_path.replace('insert_path', f'{folderpath}_{sam_input_type}')

    if object_ranking == 'random':
        for values in object_match.values():
            random.shuffle(values)
    
    idx = 0
    max_num_of_retires = min(len(object_match[llava_image_path]), 5)
    for idx in range(max_num_of_retires):
        sam_images, label_paths, cam_params_list = [], [], []
        obj_file = object_match[llava_image_path][idx]
        if 'piad' in llava_image_path:
            _, gt_afford_3d, _ = extract_point_file_piad(obj_file)
            gt_afford_3d = gt_afford_3d[:, np.argwhere(AFFORD_LIST_PIAD == afford_name).item()]
        elif 'lemon' in llava_image_path:
            _, gt_afford_3d, _ = extract_point_file_lemon(obj_file)
            gt_afford_3d = gt_afford_3d[:, np.argwhere(AFFORD_LIST_LEMON == afford_name).item()]
        else:
            print('Invalid dataset')
            return None, None, None, None, None 
        num_contact_points = np.count_nonzero(gt_afford_3d)
        if num_contact_points == 0 and is_train:
            continue
        obj_idx = obj_file.split('/')[-1].split('_')[-1][:-4]

        all_files_exist = True  # Flag to track if all required files exist
        
        sel_view_names = view_names.flatten()  # Default value
        # select random views for each row of the grid, default is set to selection based on OpenShape lookup
        if order == 'rand':
            total_sel_images = grid_size[0] * grid_size[1] * grid_size[2]
            first_half = np.random.choice(sel_view_names[:len(sel_view_names)//2], total_sel_images//2, replace=False) 
            second_half = np.random.choice(sel_view_names[len(sel_view_names)//2:], total_sel_images//2, replace=False)
            sel_view_names = np.concatenate((first_half, second_half))

        for view_name in sel_view_names:
            sam_image = os.path.join(base_rendObj_path, f'sam_{obj_name}_{obj_idx}_{afford_name}_{view_name}.png')
            label_path = os.path.join(base_rendObj_path, f'mask_{obj_name}_{obj_idx}_{afford_name}_{view_name}.png')
            sam_images.append(sam_image)
            label_paths.append(label_path)
            cam_params_list.append(normalize_cam_params(cam_params_dict[view_name]))
            if not os.path.isfile(sam_image) or not os.path.isfile(label_path):
                print(f'File does not exist: {sam_image} or {label_path}')
                all_files_exist = False
                break

        if all_files_exist:
            return sam_images, label_paths, cam_params_list, gt_afford_3d, afford_name

    print(f'\nNo valid data found after {idx+1} tries for {os.path.basename(llava_image_path)}')
    return None, None, None, None, None

class OAffordSegDataset(BaseContactSegDataset):

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
        contact_seg_data="piad_oafford||lemon_oafford",
        val_dataset=None
    ):
        super().__init__(base_image_dir, tokenizer, vision_tower, image_size)
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch if is_train else None
        self.num_classes_per_sample = num_classes_per_sample

        self.contact_seg_data = sorted(contact_seg_data.split("||") if is_train else val_dataset.split("||"))
        print(f'----> OAfford_seg_data in {"training" if is_train else "validation"}: {self.contact_seg_data}')
        self.sam_view_type = contact_dataset_config['oC_sam_view_type']
        self.view_dict = self.obj_view_dict_all[self.sam_view_type]
        self.grid_size = self.view_dict['grid_size']
        self.mask_size = (self.view_dict['mask_size'], self.view_dict['mask_size'])
        self.sam_input_type = contact_dataset_config['oC_sam_input_type']
        self.object_ranking = contact_dataset_config['oC_ranking'] if is_train else 'fixed'
        self.question_type = contact_dataset_config['oC_question_type']
        print(f'-----> contact_dataset_config: {contact_dataset_config}')

        if self.question_type == 'simple':
            self.answer_list = OAFFORD_ANSWER_LIST
            self.short_question_list = OAFFORD_QUESTION_LIST
        elif self.question_type == 'afford':
            self.answer_list = OAFFORD_AFFORD_ANSWER_LIST
            self.short_question_list = OAFFORD_AFFORD_QUESTION_LIST
        elif self.question_type == 'afford_obj':
            self.answer_list = OAFFORD_AFFORD_OBJ_ANSWER_LIST
            self.short_question_list = OAFFORD_AFFORD_QUESTION_LIST

        base_token_type = contact_dataset_config['token_type'].replace('-DifDe', '')
        if base_token_type == 'Gen-Int':
            self.answer_list = [ans.replace('OTOKEN', 'ISEG') for ans in self.answer_list]
        elif base_token_type == 'Gen-Hu-Obj':
            self.answer_list = [ans.replace('OTOKEN', 'OSEG') for ans in self.answer_list]
        else:
            self.answer_list = [ans.replace('OTOKEN', 'SEG') for ans in self.answer_list]

        self.data2list = {}
        self.data2classes = {}
        self.ds_size = {}

        for ds in self.contact_seg_data:
            split = 'Train' if is_train else 'Test'
            print(f'\n---> Initializing dataset: {ds} with split: {split}')
            classes, llava_images, object_match = init_ocontact(self.base_image_dir, ds, split=split,
                                                                ignore_keywords=self.view_dict['ignore_keywords'])
            self.data2list[ds] = (llava_images, object_match)
            self.data2classes[ds] = classes
            self.ds_size[ds] = len(llava_images)
            assert len(llava_images) == len(classes), \
                f"Mismatch in dataset {ds} llava_images: {len(llava_images)}, classes: {len(classes)}"
            print(f'---> Initialized dataset: {ds} with {len(llava_images)} images')

    def __len__(self):
        if self.is_train:
            return self.samples_per_epoch
        else:
            return sum([self.ds_size[ds] for ds in self.contact_seg_data])

    def _get_train_sample(self, idx):
        ds = random.choice(self.contact_seg_data)
        llava_images, object_match = self.data2list[ds]
        classes_list = self.data2classes[ds]
        idx = random.randint(0, len(llava_images) - 1)
        llava_image = llava_images[idx]
        sampled_classes = classes_list[idx]
        return llava_image, object_match, sampled_classes, ds
    
    def _get_val_sample(self, idx):
        total_images = 0
        for ds in self.contact_seg_data:
            ds_size = self.ds_size[ds]
            if idx < total_images + ds_size:
                llava_images, object_match = self.data2list[ds]
                classes_list = self.data2classes[ds]
                idx = idx - total_images
                break
            total_images += ds_size
        llava_image = llava_images[idx]
        sampled_classes = classes_list[idx]
        return llava_image, object_match, sampled_classes, ds 
    
    def __getitem__(self, idx):
                
        llava_image_path, object_match, sampled_classes, ds = \
            self._get_train_sample(idx) if self.is_train else self._get_val_sample(idx)

        # Get all file paths for SAM and label images
        sam_images, label_paths, cam_params, gt_afford_3d, affordance = \
                get_sam_input_and_label_oafford(self.is_train, llava_image_path, object_match, self.object_ranking, self.view_dict, self.sam_input_type)
        
        if sam_images is None: # only for training
            return self.__getitem__(0)
        
        # Load and prepare SAM images
        sam_image, valid_regions, resize = self.load_and_prepare_sam_image(sam_images, self.grid_size, self.mask_size)
        
        # Load and prepare label -- heatmap or binary mask
        if 'HM' in self.sam_view_type:
            label, contact_proportion = self.load_and_prepare_heatmap(label_paths, valid_regions, self.grid_size, self.mask_size)
        else: # Binary mask
            label, contact_proportion = self.load_and_prepare_label(label_paths, valid_regions, self.grid_size, self.mask_size)
        
        # Load and prepare CLIP image
        image_clip = self.load_and_prepare_clip_image(llava_image_path)

        # TODO: Check if we need this
        unique_label = np.unique(label).tolist()
        if self.ignore_label in unique_label:
            unique_label.remove(self.ignore_label)

        if self.is_train and len(unique_label) == 0:
            print(f'Warning: No contact pixels in the mask for {llava_image_path}')
            return self.__getitem__(random.randint(0, self.samples_per_epoch - 1))
        
        conversations, questions = self.generate_o_conversations(sampled_classes, affordance)

        cam_params = torch.stack(cam_params).float() 
        label = torch.from_numpy(label)
        masks = label.unsqueeze(1)
        gt_afford_3d = torch.from_numpy(gt_afford_3d).float()
        inference = False if self.is_train else True

        result = (
            llava_image_path,
            sam_image,      # (V, C, H, W)
            image_clip,
            conversations,
            masks,          # (V, 1, H, W)
            label[0],       # (H, W) used to get original mask
            gt_afford_3d,   # (2048,) for affordance dataset
            cam_params,     # (V, 5) normalized camera parameters
            resize,
            questions if self.is_train else None,
            sampled_classes,
            ds,
            label_paths,
        )
        if not self.is_train:
            result = result + (inference,)

        return result


class OContactSegDataset(BaseContactSegDataset):

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
        contact_seg_data="pico_ocontact",
        val_dataset=None
    ):
        super().__init__(base_image_dir, tokenizer, vision_tower, image_size)
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch if is_train else None
        self.num_classes_per_sample = num_classes_per_sample

        self.contact_seg_data = sorted(contact_seg_data.split("||") if is_train else val_dataset.split("||"))
        print(f'----> Ocontact_seg_data in {"training" if is_train else "validation"}: {self.contact_seg_data}')
        self.sam_view_type = contact_dataset_config['oC_sam_view_type']
        self.view_dict = self.obj_view_dict_all[self.sam_view_type]
        self.grid_size = self.view_dict['grid_size']
        self.mask_size = (self.view_dict['mask_size'], self.view_dict['mask_size'])
        self.sam_input_type = contact_dataset_config['oC_sam_input_type']
        self.question_type = contact_dataset_config['oC_question_type']
        print(f'-----> contact_dataset_config: {contact_dataset_config}')

        if self.question_type == 'simple':
            self.answer_list = OCONTACT_ANSWER_LIST
            self.short_question_list = OCONTACT_QUESTION_LIST
        # TODO: Check if we need to add more question types
        else:
            self.answer_list = OCONTACT_ANSWER_LIST
            self.short_question_list = OCONTACT_QUESTION_LIST

        base_token_type = contact_dataset_config['token_type'].replace('-DifDe', '')
        if base_token_type == 'Gen-Int':
            self.answer_list = [ans.replace('OTOKEN', 'ISEG') for ans in self.answer_list]
        elif base_token_type == 'Gen-Hu-Obj':
            self.answer_list = [ans.replace('OTOKEN', 'OSEG') for ans in self.answer_list]
        else:
            self.answer_list = [ans.replace('OTOKEN', 'SEG') for ans in self.answer_list]

        self.data2list = {}
        self.data2classes = {}
        self.ds_size = {}

        for ds in self.contact_seg_data:
            split = 'Train' if is_train else 'Test'
            print(f'\n---> Initializing dataset: {ds} with split: {split}')
            classes, llava_images, object_match = init_ocontact(self.base_image_dir, ds, split=split,
                                                                ignore_keywords=self.view_dict['ignore_keywords'])
            self.data2list[ds] = (llava_images, object_match)
            self.data2classes[ds] = classes
            self.ds_size[ds] = len(llava_images)
            assert len(llava_images) == len(classes), \
                f"Mismatch in dataset {ds} llava_images: {len(llava_images)}, classes: {len(classes)}"
            print(f'---> Initialized dataset: {ds} with {len(llava_images)} images')

    def __len__(self):
        if self.is_train:
            return self.samples_per_epoch
        else:
            return sum([self.ds_size[ds] for ds in self.contact_seg_data])

    def _get_train_sample(self, idx):
        ds = random.choice(self.contact_seg_data)
        llava_images, object_match = self.data2list[ds]
        classes_list = self.data2classes[ds]
        idx = random.randint(0, len(llava_images) - 1)
        llava_image = llava_images[idx]
        sampled_classes = classes_list[idx]
        return llava_image, object_match, sampled_classes, ds
    
    def _get_val_sample(self, idx):
        total_images = 0
        for ds in self.contact_seg_data:
            ds_size = self.ds_size[ds]
            if idx < total_images + ds_size:
                llava_images, object_match = self.data2list[ds]
                classes_list = self.data2classes[ds]
                idx = idx - total_images
                break
            total_images += ds_size
        llava_image = llava_images[idx]
        sampled_classes = classes_list[idx]
        return llava_image, object_match, sampled_classes, ds 
    
    def __getitem__(self, idx):
                
        llava_image_path, object_match, sampled_classes, ds = \
            self._get_train_sample(idx) if self.is_train else self._get_val_sample(idx)

        # Get all file paths for SAM and label images
        sam_images, label_paths, cam_params, gt_contact_3d, affordance = \
            get_sam_input_and_label_ocontact(llava_image_path, object_match, self.view_dict)

        if sam_images is None: # only for training
            return self.__getitem__(0)
        
        # Load and prepare SAM images
        sam_image, valid_regions, resize = self.load_and_prepare_sam_image(sam_images, self.grid_size, self.mask_size)
        
        # Load and prepare label -- heatmap or binary mask
        label, contact_proportion = self.load_and_prepare_label(label_paths, valid_regions, self.grid_size, self.mask_size)
        
        # Load and prepare CLIP image
        image_clip = self.load_and_prepare_clip_image(llava_image_path)

        # TODO: Check if we need this
        unique_label = np.unique(label).tolist()
        if self.ignore_label in unique_label:
            unique_label.remove(self.ignore_label)

        if self.is_train and len(unique_label) == 0:
            print(f'Warning: No contact pixels in the mask for {llava_image_path}')
            return self.__getitem__(random.randint(0, self.samples_per_epoch - 1))
        
        conversations, questions = self.generate_o_conversations(sampled_classes, affordance)

        cam_params = torch.stack(cam_params).float() 
        label = torch.from_numpy(label)
        masks = label.unsqueeze(1)
        gt_contact_3d = torch.from_numpy(gt_contact_3d).float()
        inference = False if self.is_train else True

        result = (
            llava_image_path,
            sam_image,      # (V, C, H, W)
            image_clip,
            conversations,
            masks,          # (V, 1, H, W)
            label[0],       # (H, W) used to get original mask
            gt_contact_3d,   # (N,) for pico dataset
            cam_params,     # (V, 5) normalized camera parameters
            resize,
            questions if self.is_train else None,
            sampled_classes,
            ds,
            label_paths,
        )
        if not self.is_train:
            result = result + (inference,)

        return result
