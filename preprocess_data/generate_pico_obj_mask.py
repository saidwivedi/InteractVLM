import os
import numpy as np
import random
import tqdm
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import cv2
import joblib as jl
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
)

from render_mesh_utils import compute_vertex_normals
from render_mesh_utils import render_mesh, project_vertices_and_create_mask, verify_contact_reconstruction_diff

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



PICO_DATA_ROOT = f'/is/cluster/fast/groups/ps-pico/images_with_contact'
PICO_CONTACT_MIDPOLY_MESH = '/is/cluster/fast/sdwivedi/work/lemon_3d/Data/pico/midpoly_contact_jsons'
PICO_TRAIN_LIST = f'/is/cluster/fast/sdwivedi/work/lemon_3d/Data/damon/train/imgname.npy'
PICO_TEST_LIST = f'/is/cluster/fast/sdwivedi/work/lemon_3d/Data/damon/test/imgname.npy'
PICO_OUTDIR = f'/is/cluster/fast/sdwivedi/work/lemon_3d/Data/pico'

debug = False


LIGHT_LOCATIONS = [[0, 0, 3], [0, 0, 3], [0, 0, -3], [0, 0, -3]]
RENDER_IMG_SIZE = (1024, 1024)

VIEWS4 = {
    'frontleft': (1.5, 45, 315, 0., 0.0),
    'frontright': (1.5, 45, 45, 0., 0.0),
    'backleft': (1.5, 330, 135, 0., 0.0),
    'backright': (1.5, 330, 225, 0., 0.0),
}
VIEWS6 = {
    'front': (2, 0, 180, 0., 0.3),
    'back': (2, 0, 0, 0., 0.3),
    'left': (2, 0, 90, 0., 0.3),
    'right': (2, 0, 270, 0., 0.3),
    'top': (2, 90, 0, 0., 0.3),
    'bottom': (2, 270, 0, 0., 0.3),
}

def normalize_mesh(vertices, scale_factor=1.0):
    
    # Center the mesh
    centroid = torch.mean(vertices, dim=0)
    vertices_centered = vertices - centroid
    
    # Get the bounding box dimensions
    bbox_min = torch.min(vertices_centered, dim=0)[0]
    bbox_max = torch.max(vertices_centered, dim=0)[0]
    bbox_sizes = bbox_max - bbox_min
    
    # Scale by the largest dimension to maintain aspect ratio
    norm_scale = torch.max(bbox_sizes)
    vertices_normalized = vertices_centered / norm_scale * scale_factor
    
    return vertices_normalized

def get_contact_vertex_indices(obj_mesh, contact_points):
    contact_vertex_indices = []
    for cp in contact_points:
        if cp.startswith("f "):
            parts = cp.split()
            face_idx = int(parts[1])

            # Get all vertices of the face
            v0, v1, v2 = obj_mesh.faces[face_idx]
            contact_vertex_indices.extend([v0, v1, v2])

    # Remove duplicates
    contact_vertex_indices = list(set(map(int, contact_vertex_indices)))
    return contact_vertex_indices
    
def get_contact_points(obj_mesh, contact_annot_f):
    with open(contact_annot_f, 'r') as f:
        contact_data = json.load(f)

    contact_dict = {}
    for entry in contact_data["data"]:
        name = entry.get("name", "")
        if name.startswith("objShape"):
            contact_points = entry.get("contactPoints", [])
            vertex_indices = get_contact_vertex_indices(obj_mesh, contact_points)
            name = name.replace("objShape", "")
            contact_dict[name] = vertex_indices

    return contact_dict

def generate_object_mask(imgname, mesh, contact_dict, out_dir, views_dict, debug=False):

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)

    vertices = normalize_mesh(vertices)
    min_coords, _ = torch.min(vertices, dim=0)
    max_coords, _ = torch.max(vertices, dim=0)
    vertex_colors_norm = (vertices - min_coords) / (max_coords - min_coords)
    vertex_colors_norm = vertex_colors_norm * 0.8 + 0.1

    mesh_color = Meshes(verts=[vertices],
            faces=[faces],
            textures=TexturesVertex(verts_features=vertex_colors_norm.unsqueeze(0).to(device)))

    all_contact_vertices = set()
    for indices in contact_dict.values():
        all_contact_vertices.update(indices)
    contact_vertices_flat = list(all_contact_vertices)

    mask_list, render_list = [], []
    pixel_to_vertices_map_list, bary_coords_list = [], []

    for idx, (save_str, camera_params) in enumerate(views_dict.items()):

        sam_path = os.path.join(out_dir, f'sam_{imgname}_{save_str}.png')
        mask_path = os.path.join(out_dir, f'mask_{imgname}_{save_str}.png')
        if os.path.exists(mask_path) and os.path.exists(sam_path):
            print(f'already exists {mask_path}')
            continue
        
        render_rgb = render_mesh(mesh_color, camera_params, LIGHT_LOCATIONS[idx], RENDER_IMG_SIZE)

        mask, pixel_to_vertices_map, bary_coords = \
                project_vertices_and_create_mask(mesh_color, camera_params, contact_vertices_flat, RENDER_IMG_SIZE, min_vertices=3)
        mask = (mask * 255).astype(np.uint8)

        cv2.imwrite(sam_path, cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        np.savez_compressed(os.path.join(out_dir, f'p2vmap_{imgname}_{save_str}.npz'), 
                            pixel_to_vertices_map=pixel_to_vertices_map, 
                            bary_coords_map=bary_coords,
                            num_vertices=vertices.shape[0])
        
        if debug:
            mask_list.append(mask)
            render_list.append(render_rgb)
            pixel_to_vertices_map_list.append(pixel_to_vertices_map)
            bary_coords_list.append(bary_coords)

    # Save the contact vertices
    contact_vertices = np.zeros((vertices.shape[0],), dtype=np.float32)
    contact_vertices[contact_vertices_flat] = 1.0
    # print(f"Number of contact vertices: {len(contact_vertices_flat)}")
    # print(f"num of non-zero contact vertices: {len(np.where(contact_vertices > 0)[0])}")
    jl.dump(contact_vertices, os.path.join(out_dir, f'contact_vertices_{imgname}.pkl'))

    if debug:
        verify_contact_reconstruction_diff(mask_list, pixel_to_vertices_map_list, \
                                           bary_coords_list, contact_vertices_flat, \
                                           threshold=0.5, debug=debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or val')
    parser.add_argument('--view_type', type=str, default='views4', help='view type')
    parser.add_argument('--output_dir', type=str, default='midpoly_mesh_0507', help='output directory')
    parser.add_argument('--contact_resolution', type=str, default='mid', help='contact resolution')
    
    args = parser.parse_args()

    if args.split == 'train':
        pico_list = np.load(PICO_TRAIN_LIST, allow_pickle=True)
    else:
        pico_list = np.load(PICO_TEST_LIST, allow_pickle=True)

    pico_list = [os.path.basename(x) for x in pico_list]
    print(f"Number of samples in {args.split} list: {len(pico_list)}")

    views_dict = VIEWS6 if args.view_type == 'views6' else VIEWS4
    imglist = []

    for obj_cat in tqdm.tqdm(sorted(os.listdir(PICO_DATA_ROOT))):
        # if obj_cat != 'chair':
        #     continue
        obj_cat_dir = os.path.join(PICO_DATA_ROOT, obj_cat)

        for idx, obj_sample in enumerate(sorted(os.listdir(obj_cat_dir))):

            if obj_sample not in pico_list:
                print(f"Skipping {obj_sample} as it is not in the list.")
                continue

            obj_sample_dir = os.path.join(obj_cat_dir, obj_sample)
            if args.contact_resolution == 'low':
                contact_annot_f = os.path.join(obj_sample_dir, 'corresponding_contacts.json')
            elif args.contact_resolution == 'mid':
                contact_annot_f = os.path.join(PICO_CONTACT_MIDPOLY_MESH, 
                f'{obj_cat}__{obj_sample}__corresponding_contacts_MIDPOLY.json')

            outdir = os.path.join(PICO_OUTDIR, args.split, args.output_dir, obj_cat)
            os.makedirs(outdir, exist_ok=True)

            if not os.path.exists(contact_annot_f):
                continue
            
            img_fname = f"{obj_cat}__{obj_sample}"
            img_f = f"{obj_sample_dir}/{img_fname}"
            os.system(f"cp {img_f} {PICO_OUTDIR}/{args.split}/images/")
            # os.system(f"cp {img_f} {outdir}")

            contact_dict = None

            if args.contact_resolution == 'low':
                obj_mesh_f = os.path.join(obj_sample_dir, 'object_OpenShape_selected_lowpoly.obj')
            elif args.contact_resolution == 'mid':
                obj_mesh_f = os.path.join(obj_sample_dir, 'object_OpenShape_selected.obj')

            if not os.path.exists(obj_mesh_f):
                print(f"Mesh file {obj_mesh_f} does not exist, skipping {obj_sample}.")
                continue
            
            imglist.append(img_fname)

            obj_mesh = trimesh.load_mesh(obj_mesh_f,
                                        process=False,
                                        force='mesh',  # Force loading as a simple mesh
                                        ignore_materials=True)
            
            print("length of vertices: ", len(obj_mesh.vertices))
            
            contact_dict = get_contact_points(obj_mesh, contact_annot_f)
            generate_object_mask(obj_sample[:-4], obj_mesh, contact_dict, outdir, views_dict, debug=debug)

            print(f"Generated object mask for {obj_sample} in {outdir}")

    print(f"Total number of images processed: {len(imglist)}")
    jl.dump(imglist, os.path.join(PICO_OUTDIR, args.split, f'{args.split}_imglist_{args.contact_resolution}poly.pkl'))