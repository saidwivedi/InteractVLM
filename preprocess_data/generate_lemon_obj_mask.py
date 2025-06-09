import numpy as np
import torch
import os
import random
import cv2
import argparse

from pytorch3d.structures import Pointclouds
from utils_obj_pc import (
    enhance_point_cloud_structure_preserving, get_pc_color_by_depth, get_pc_color_by_position,
    render_pc_p3d, smooth_mask, create_affordance_mask, normalize_point_cloud
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
    parser.add_argument('--split', type=str, default='val', help='train or val')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--view_type', type=str, default='views_4', help='2 or 4 or 16 views')

    args = parser.parse_args()

    print(f'Generating lemon object masks for {args.split} split and saving to {args.output_dir} directory')

    lemon_obj_affordance_dict = {}
    split = args.split
    device = torch.device("cuda:0")
    RENDER_SIZE = (1024, 1024)
    DYNAMIC_RADIUS = True
    NUM_POINT2PIXEL = 1
    KERNEL_SIZE = 20
    ROOT_DIR = f'./data'
    IMG_FOLDER_LEMON = f'{ROOT_DIR}/lemon/Images/'
    OUTPUT_DIR = f'{ROOT_DIR}/lemon/lemon_ocontact/{args.output_dir}/'

    # 'name': (distance, elevation, azimuth, x_trans, y_trans)
    views_2 = {
        'frontleft': (2, 45, -30, 0., 0.5),
        'frontright': (2, 45, 30, 0., 0.5),
    }
    views_4 = {
        'frontleft': (2, 45, 315, 0., 0.0),
        'frontright': (2, 45, 45, 0., 0.0),
        'backleft': (2, 330, 135, 0., 0.0),
        'backright': (2, 330, 225, 0., 0.0)
    }
    views_6 = {
        'top': (2, 90, 0, 0., 0.0),
        'bottom': (2, 270, 0, 0., 0.0),
        'front': (2, 0, 0, 0., 0.0),
        'back': (2, 0, 180, 0., 0.0),
        'left': (2, 0, 270, 0., 0.0),
        'right': (2, 0, 90, 0., 0.0)
    }
    views = eval(args.view_type)

    for obj in sorted(os.listdir(IMG_FOLDER_LEMON)):
        afford_list = os.listdir(os.path.join(IMG_FOLDER_LEMON, obj))
        lemon_obj_affordance_dict[obj] = afford_list

    print(lemon_obj_affordance_dict)

    OBJ_LIST_LEMON = f'{ROOT_DIR}/lemon/txt_scripts/Point_{split}.txt'
    objs_lemon = open(OBJ_LIST_LEMON, 'r').read().split('\n')

    count, max_count = 0, -1
    total_objs = len(objs_lemon)
    disp_img_list = []

    random.shuffle(objs_lemon)

    # run over all objects
    for obj_f in objs_lemon:

        if count >= max_count and max_count > 0:
                break

        pc_cord, afford_dict, obj_name = extract_point_file_lemon(obj_f)
        pc_cord = normalize_point_cloud(pc_cord)

        if obj_name not in AFFORD_PROB_LEMON.keys():
            continue

        # for each affordance of that object
        for afford_label in lemon_obj_affordance_dict[obj_name]:

            if afford_label not in list(AFFORD_PROB_LEMON[obj_name].keys()):
                continue

            save_dir = os.path.join(OUTPUT_DIR, obj_name, afford_label)
            img_name = obj_f.split('/')[-1][:-4]
            os.makedirs(save_dir, exist_ok=True)

            try:
                afford_prob = AFFORD_PROB_LEMON[obj_name][afford_label]
                print(f'{count}/{total_objs} Processing object: {obj_name}, affordance: {afford_label} with prob: {afford_prob}')
                # get the affordance point cloud
                afford_pc = np.argwhere(afford_dict[:, np.argwhere(AFFORD_LIST_LEMON == afford_label).item()] > afford_prob).squeeze()
                # enhance the point cloud to get more points
                dense_points, dense_afford_pc = enhance_point_cloud_structure_preserving(pc_cord, afford_pc, target_num_points=100000)
                # convert to torch tensors
                verts = torch.Tensor(dense_points).to(device)
                
                # get point_cloud color based on depth or position
                if 'color' in OUTPUT_DIR:
                    pc_rgb_color = get_pc_color_by_position(dense_points)
                else:
                    pc_rgb_color = get_pc_color_by_depth(dense_points)
                
                # create point clouds
                dense_point_cloud_color = Pointclouds(points=[verts], features=[pc_rgb_color])

                for save_str, camera_params in views.items():
                    image_color = render_pc_p3d(dense_point_cloud_color, camera_params,\
                                               dynamic_radius=DYNAMIC_RADIUS, \
                                               image_size=RENDER_SIZE)
                    mask, pixel_to_point_map  = create_affordance_mask(dense_point_cloud_color, dense_afford_pc, camera_params, \
                                                                       dynamic_radius=DYNAMIC_RADIUS, image_size=RENDER_SIZE, \
                                                                       num_point2pixel=NUM_POINT2PIXEL)
                    blurred_mask = smooth_mask(mask, kernel_size=KERNEL_SIZE)

                    cv2.imwrite(os.path.join(save_dir, f'sam_{img_name}_{afford_label}_{save_str}.png'), cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_dir, f'mask_{img_name}_{afford_label}_{save_str}.png'), blurred_mask)
                    np.savez_compressed(os.path.join(save_dir, f'p2pmap_{img_name}_{afford_label}_{save_str}'), \
                                        mapping=pixel_to_point_map)
                    
                count += 1

            except Exception as e:
                print(e)
                continue