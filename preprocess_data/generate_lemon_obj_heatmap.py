import numpy as np
import torch
import os
import random
import cv2
import argparse

from pytorch3d.structures import Pointclouds
from utils_obj_pc import (
    get_pc_color_by_depth, get_pc_color_by_position,
    render_pc_p3d, smooth_mask, create_affordance_heatmap, normalize_point_cloud
)
from constants import AFFORD_LIST_LEMON, AFFORD_PROB_LEMON

def extract_point_file_lemon(path):
    with open(path,'r') as f:
        coordinates = []
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.strip(' ')
        data = line.split(' ')
        coordinate = [float(x) for x in data]
        coordinates.append(coordinate)
    data_array = np.array(coordinates)
    points_coordinates = data_array[:, 0:3]
    affordance_label = data_array[: , 3:]
    obj_name = path.split('/')[-1].split('_')[0]

    return points_coordinates, affordance_label, obj_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='Train', help='train or val')
    parser.add_argument('--output_dir', type=str, default='output_color', help='output directory')
    parser.add_argument('--view_type', type=str, default='views_4', help='2 or 4 or 6 or 8 or 10')
    parser.add_argument('--run_id', type=int, default='0', help='0 to 10 for different views')
    parser.add_argument('--sam_input_type', type=str, default='color', help='color or depth')

    args = parser.parse_args()

    lemon_obj_affordance_dict = {}
    split = args.split
    device = torch.device("cuda:0")
    RENDER_SIZE = (1024, 1024)
    DYNAMIC_RADIUS = False
    NUM_POINT2PIXEL = 1
    FIXED_RADIUS = 0.03
    KERNEL_SIZE = 20
    ROOT_DIR = f'./data'
    IMG_FOLDER_LEMON = f'{ROOT_DIR}/lemon/Images/'
    OUTPUT_DIR = f'{ROOT_DIR}/lemon/lemon_ocontact/{args.output_dir}/'

    # 'name': (distance, elevation, azimuth, x_trans, y_trans)
    views_dict = {
        'frontleft': (1.5, 45, 315, 0., 0.0),
        'frontright': (1.5, 45, 45, 0., 0.0),
        'backleft': (1.5, 330, 135, 0., 0.0),
        'backright': (1.5, 330, 225, 0., 0.0),
        'top': (1.5, 90, 0, 0., 0.0),
        'bottom': (1.5, 270, 0, 0., 0.0),
        'front': (1.5, 0, 0, 0., 0.0),
        'back': (1.5, 0, 180, 0., 0.0),
        'left': (1.5, 0, 270, 0., 0.0),
        'right': (1.5, 0, 90, 0., 0.0)
    }  

    views = list(views_dict.items())[args.run_id:args.run_id+1]

    for obj in sorted(os.listdir(IMG_FOLDER_LEMON)):
        afford_list = os.listdir(os.path.join(IMG_FOLDER_LEMON, obj))
        lemon_obj_affordance_dict[obj] = afford_list

    print(lemon_obj_affordance_dict)

    OBJ_LIST_LEMON = f'{ROOT_DIR}/lemon/txt_scripts/Point_{split}.txt'
    objs_lemon = open(OBJ_LIST_LEMON, 'r').read().split('\n')

    count, max_count = 0, -1
    total_objs = len(objs_lemon)
    disp_img_list = []

    # random.shuffle(objs_lemon)

    # run over all objects
    for obj_f in objs_lemon:

        if count >= max_count and max_count > 0:
                break

        pc_cord, afford_dict, obj_name = extract_point_file_lemon(obj_f)
        pc_cord = normalize_point_cloud(pc_cord)


        # for each affordance of that object
        for afford_label in lemon_obj_affordance_dict[obj_name]:
            
            save_dir = os.path.join(OUTPUT_DIR, obj_name, afford_label)
            img_name = obj_f.split('/')[-1][:-4]
            os.makedirs(save_dir, exist_ok=True)
            try:
                print(f'\n{count}/{total_objs} Processing object: {obj_name}, affordance: {afford_label}')
                # get the affordance point cloud
                afford_pc = afford_dict[:, np.argwhere(AFFORD_LIST_LEMON == afford_label).item()]
                # convert to torch tensors
                verts = torch.Tensor(pc_cord).to(device)
                
                # get point_cloud color based on depth or position
                if args.sam_input_type == 'color':
                    pc_rgb_color = get_pc_color_by_position(pc_cord)
                else:
                    pc_rgb_color = get_pc_color_by_depth(pc_cord)
                
                # create point clouds
                point_cloud_color = Pointclouds(points=[verts], features=[pc_rgb_color])

                for save_str, camera_params in views:
                    mask_path = os.path.join(save_dir, f'mask_{img_name}_{afford_label}_{save_str}.png')
                    sam_path = os.path.join(save_dir, f'sam_{img_name}_{afford_label}_{save_str}.png')
                    if os.path.exists(mask_path) and os.path.exists(sam_path):
                        print(f'already exists {mask_path}')
                        continue
                    print(f'saving to {mask_path}')
                    image_color = render_pc_p3d(point_cloud_color, camera_params,\
                                                dynamic_radius=DYNAMIC_RADIUS, fixed_radius=FIXED_RADIUS, \
                                                image_size=RENDER_SIZE)
                    mask, pixel_to_point_map  = create_affordance_heatmap(point_cloud_color, afford_pc, camera_params, \
                                                                            dynamic_radius=DYNAMIC_RADIUS, fixed_radius=FIXED_RADIUS, \
                                                                            image_size=RENDER_SIZE, num_point2pixel=NUM_POINT2PIXEL)

                    cv2.imwrite(sam_path, cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                    np.savez_compressed(os.path.join(save_dir, f'p2pmap_{img_name}_{afford_label}_{save_str}'), \
                                        mapping=pixel_to_point_map)

                count += 1

            except Exception as e:
                print(e)
                continue