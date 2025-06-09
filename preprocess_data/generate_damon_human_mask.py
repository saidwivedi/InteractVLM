import os
import numpy as np
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import cv2
import joblib as jl
from smplx import build_layer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
)

from render_mesh_utils import compute_vertex_normals, get_body_params
from render_mesh_utils import render_mesh, project_vertices_and_create_mask, verify_contact_reconstruction_diff
from render_mesh_utils import VIRTUVIAN_POSE

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMPLX_BODY_MODEL_PATH = 'data/body_models/smplx'
SMPLH_BODY_MODEL_PATH = 'data/body_models/smplh'
SMPL_BODY_MODEL_PATH = 'data/body_models/smpl'


#### SMPL-H ####
body_model = build_layer(
                model_path = f'{SMPLH_BODY_MODEL_PATH}/SMPLH_NEUTRAL.pkl',
                model_type = "smplh",
                use_pca = False,
                gender = 'neutral',
                ext = 'pkl',
                num_betas=10,
    )
BODY_FACES = torch.from_numpy(body_model.faces.astype(np.int32))

DAMON_DATA_ROOT = './data/damon/'
DAMON_TRAIN_LIST = f'{DAMON_DATA_ROOT}/train/imgname.npy'
DAMON_TRAIN_IMG_ROOT = f'{DAMON_DATA_ROOT}/train/images'
DAMON_TRAIN_CONTACT = f'{DAMON_DATA_ROOT}/train/contact_label.npy'
DAMON_TRAIN_CONTACT_OBJ = f'{DAMON_DATA_ROOT}/train/contact_label_objectwise.npy'

DAMON_TEST_LIST = f'{DAMON_DATA_ROOT}/test/imgname.npy'
DAMON_TEST_IMG_ROOT = f'{DAMON_DATA_ROOT}/test/images'
DAMON_TEST_CONTACT = f'{DAMON_DATA_ROOT}/test/contact_label.npy'
DAMON_TEST_CONTACT_OBJ = f'{DAMON_DATA_ROOT}/test/contact_label_objectwise.npy'

debug = False


LIGHT_LOCATIONS = [[0, 0, 3], [0, 0, -3], [0, 0, -3], [0, 0, 3]]
RENDER_IMG_SIZE = (1024, 1024)

VIEWS4 = {
    'topfront': (2, 45, 315, 0., 0.0),
    'topback': (2, 45, 135, 0., 0.0),
    'bottomfront': (2, 315, 315, 0., 0.3),
    'bottomback': (2, 315, 135, 0., 0.3),
}
VIEWS6 = {
    'front': (2, 0, 180, 0., 0.3),
    'back': (2, 0, 0, 0., 0.3),
    'left': (2, 0, 90, 0., 0.3),
    'right': (2, 0, 270, 0., 0.3),
    'top': (2, 90, 0, 0., 0.3),
    'bottom': (2, 270, 0, 0., 0.3),
}

MERGED_SEGM = jl.load('./data/smpl_segmentation_merged.pkl')

def get_body_parts_from_vertices(vertices_list, threshold=0.1):

    # merged_segm = jl.load('multiview_visualization/smpl_segmentation_multiview.pkl')
    body_parts = []
    
    # Convert input vertices to set for faster lookup
    vertices_set = set(vertices_list)
    
    # Check each body part
    for part, part_vertices in MERGED_SEGM.items():
        # Convert part vertices to set
        part_vertices_set = set(part_vertices)
        
        # Calculate coverage
        intersection = len(vertices_set.intersection(part_vertices_set))
        coverage = intersection / len(part_vertices_set)
        
        # If coverage exceeds threshold, add to result
        if coverage >= threshold:
            body_parts.append(part)
            
    return body_parts

def generate_human_mask(imgname, mesh, contact_vertices, out_dir, views_dict, debug=False):
    mask_list, render_list, pixel_to_vertices_map_list, bary_coords_list = [], [], [], []

    for idx, (save_str, camera_params) in enumerate(views_dict.items()):
        save_path = f'{out_dir}/{imgname[:-4]}_{save_str}.png'
        if os.path.exists(save_path):
            continue
        
        mask, pixel_to_vertices_map, bary_coords = project_vertices_and_create_mask(
            mesh, camera_params, contact_vertices, image_size=RENDER_IMG_SIZE)
        
        ### no need to render for all images -- fixed for all images ###
        render = render_mesh(mesh, camera_params, LIGHT_LOCATIONS[idx], image_size=RENDER_IMG_SIZE)

        cv2.imwrite(save_path, mask)
        
        if debug:
            np.savez_compressed(f'{out_dir}/v2pmap_{imgname[:-4]}_{save_str}.npz', 
                            pixel_to_vertices_map=pixel_to_vertices_map, 
                            bary_coords=bary_coords)
            mask_list.append(mask)
            pixel_to_vertices_map_list.append(pixel_to_vertices_map)
            bary_coords_list.append(bary_coords)
            render_list.append(render)

    if debug:
        # Verify contact reconstruction
        verify_contact_reconstruction_diff(mask_list, pixel_to_vertices_map_list, bary_coords_list, contact_vertices, 
                                           threshold=0.5, debug=debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or val')
    parser.add_argument('--mask_type', type=str, default='objectwise', help='all contact or objectwise')
    parser.add_argument('--output_dir', type=str, default='test', help='output directory')
    parser.add_argument('--view_type', type=str, default='views4', help='view type')
    
    args = parser.parse_args()

    if args.split == 'train':
        damon_list = np.load(DAMON_TRAIN_LIST, allow_pickle=True)
        damon_contact = np.load(DAMON_TRAIN_CONTACT, allow_pickle=True)
        damon_contact_obj = np.load(DAMON_TRAIN_CONTACT_OBJ, allow_pickle=True)
    else:
        damon_list = np.load(DAMON_TEST_LIST, allow_pickle=True)
        damon_contact = np.load(DAMON_TEST_CONTACT, allow_pickle=True)
        damon_contact_obj = np.load(DAMON_TEST_CONTACT_OBJ, allow_pickle=True)

    views_dict = VIEWS6 if args.view_type == 'views6' else VIEWS4


    ######## vertices and pose are fixed for all images ########
    body = body_model(body_pose=VIRTUVIAN_POSE)
    vertices = body.vertices[0].detach()
    vertex_normals = compute_vertex_normals(vertices, BODY_FACES)
    vertex_colors = (vertex_normals + 1) / 2
    mesh = Meshes(verts=[vertices.to(device)], 
                  faces=[BODY_FACES.to(device)],
                  textures=TexturesVertex(verts_features=vertex_colors.unsqueeze(0).to(device)))
    #############################################################

    missing_contact = {} 
    total_valid_annotations = 0
    body_parts_name = {}

    for idx, img_f in enumerate(damon_list):

        imgname = os.path.basename(img_f)
        root_folder = eval(f'DAMON_{args.split.upper()}_IMG_ROOT')
        img_f = f'{root_folder}/{imgname}'
        
        rgb_img = Image.open(img_f)

        body = body_model(body_pose=VIRTUVIAN_POSE)
        vertices = body.vertices[0].detach()
        
        # create mask for each object
        if args.mask_type == 'objectwise':
            for obj in list(damon_contact_obj[idx].keys()):
                contact_vertices = damon_contact_obj[idx][obj]
                out_dir = f'{DAMON_DATA_ROOT}/{args.split}/{args.output_dir}/{obj}'
                if len(contact_vertices) == 0:
                    if obj not in missing_contact.keys():
                        missing_contact[obj] = 0
                    missing_contact[obj] += 1
                    # print(f'Skipping {idx}/{len(damon_list)} {imgname} {obj} because of no contact vertices')
                    continue
                part_names = get_body_parts_from_vertices(contact_vertices)
                print(f'{imgname}_{obj} | Contact vertices: {len(contact_vertices)} | Body parts: {part_names}')
                body_parts_name[f'{imgname[:-4]}_{obj}'] = part_names
                os.makedirs(out_dir, exist_ok=True)
                print(f'processing {idx}/{len(damon_list)} {imgname} | Contact vertices: {len(contact_vertices)} \nMissing: {missing_contact}')
                # rgb_img.save(f'{out_dir}/{imgname}')
                generate_human_mask(imgname, mesh, contact_vertices, out_dir, views_dict, debug=debug)

        # create mask for all contact vertices
        else:
            contact_vertices = np.where(damon_contact[idx] != 0)[0]
            out_dir = f'{DAMON_DATA_ROOT}/{args.split}/{args.output_dir}/all_contact'
            if len(contact_vertices) == 0:
                if 'all_contact' not in missing_contact.keys():
                    missing_contact['all_contact'] = 0
                missing_contact['all_contact'] += 1
                print(f'Skipping {idx}/{len(damon_list)} {imgname} because of no contact vertices')
                continue
            part_names = get_body_parts_from_vertices(contact_vertices)
            print(f'{imgname} | Contact vertices: {len(contact_vertices)} | Body parts: {part_names}')
            body_parts_name[f'{imgname[:-4]}'] = part_names
            os.makedirs(out_dir, exist_ok=True)
            print(f'processing {idx}/{len(damon_list)} {imgname} | Contact vertices: {len(contact_vertices)} \nMissing: {missing_contact}')
            # # rgb_img.save(f'{out_dir}/{imgname}')
            generate_human_mask(imgname, mesh, contact_vertices, out_dir, views_dict, debug=debug)

        total_valid_annotations += 1

    jl.dump(body_parts_name, f'{DAMON_DATA_ROOT}/{args.split}/body_parts_{args.mask_type}.pkl')
    
    print(f'Total valid annotations: {total_valid_annotations}/{len(damon_list)}')



        

        


