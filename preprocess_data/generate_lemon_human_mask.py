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

#### SMPL-X ####
# body_model = SMPLX(SMPLX_BODY_MODEL_PATH, ext='pkl')
# body_faces = torch.from_numpy(body_model.faces.astype(np.int32))

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

DATA_ROOT = './data/lemon/'
SMPLH_PARAM_FILE = f'{DATA_ROOT}/smplh_param/human_label.json'
SMPLH_CONTACT_ROOT = f'{DATA_ROOT}/smplh_contact_pkl'

TRAIN_LIST = f'{DATA_ROOT}/txt_scripts/train.txt'
VAL_LIST = f'{DATA_ROOT}/txt_scripts/val.txt'

debug = False

LEMON_SMPLH_DATA = json.load(open(SMPLH_PARAM_FILE, 'r'))


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='train or val')
    parser.add_argument('--output_dir', type=str, default='test', help='output directory')
    parser.add_argument('--view_type', type=str, default='views4', help='view type')
    
    args = parser.parse_args()

    if args.split == 'train':
        images = open(TRAIN_LIST, 'r').read().splitlines()
    else:
        images = open(VAL_LIST, 'r').read().splitlines()

    views_dict = VIEWS6 if args.view_type == 'views6' else VIEWS4

    random.shuffle(images)

    ######## vertices and pose are fixed for all images ########
    body = body_model(body_pose=VIRTUVIAN_POSE)
    vertices = body.vertices[0].detach()
    vertex_normals = compute_vertex_normals(vertices, BODY_FACES)
    vertex_colors = (vertex_normals + 1) / 2
    mesh = Meshes(verts=[vertices.to(device)], 
                  faces=[BODY_FACES.to(device)],
                  textures=TexturesVertex(verts_features=vertex_colors.unsqueeze(0).to(device)))
    ############################################################# 

    body_parts_name = {}
    no_contact_count = 0

    for idx, img_f in enumerate(images):
        sample_name = '/'.join(img_f.split('/')[-3:])
        print(f'processing {idx}/{len(images)} {sample_name}')
        out_dir = f'{DATA_ROOT}/{args.output_dir}/{os.path.dirname(sample_name)}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        key = f'Images/{sample_name}'
        imgname = os.path.basename(img_f)
        rgb_img = Image.open(img_f)

        # Load contact information
        smplh_contact = jl.load(f'{SMPLH_CONTACT_ROOT}/{sample_name[:-4]}.pkl')
        contact_vertices = np.where(smplh_contact != 0)[0]
        
        part_names = get_body_parts_from_vertices(contact_vertices)
        if len(part_names) == 0:
            print(f'No contact vertices found for {imgname}')
            no_contact_count += 1
        body_parts_name[imgname[:-4]] = part_names

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
            reconstructed_contact, missed_vertices, extra_vertices = verify_contact_reconstruction_diff(
                                    mask_list, pixel_to_vertices_map_list, bary_coords_list, contact_vertices, threshold=0.5)

    jl.dump(body_parts_name, f'{DATA_ROOT}/body_parts_{args.split}.pkl')

    print(f'No contact count: {no_contact_count}')

        


