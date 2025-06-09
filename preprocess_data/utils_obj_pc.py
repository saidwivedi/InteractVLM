import numpy as np
import torch
import cv2
from scipy.spatial import cKDTree

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    NormWeightedCompositor,
)


######################################################################################################################
def get_dynamic_radius(pc):
    point_coords = pc.points_padded().cpu().numpy()  # Extract points
    bbox_min = point_coords.min(axis=1)  # Bounding box min
    bbox_max = point_coords.max(axis=1)  # Bounding box max
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min, axis=1)  # Diagonal distance of bounding box
    # for 512x512, 0.003 is a good scaling factor
    # for 1024x1024, 0.004 is a good scaling factor
    radius = 0.004 * bbox_diagonal.mean()
    return radius

def get_rasterizer(camera_params, radius, image_size=(512, 512), device='cuda'):
    distance, elevation, azimuth, x_trans, y_trans = camera_params
    R, T = look_at_view_transform(distance, elevation, azimuth)
    T[0, 1] += y_trans
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=10,
        max_points_per_bin=50000,
        bin_size=None,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


####################################### Functions for creating affordance masks #######################################

def lift_masks_to_pointcloud(masks_list, pixel_to_point_maps_list, num_points, num_point2pixel=1):

    point_votes = np.zeros(num_points, dtype=np.float32)
    point_counts = np.zeros(num_points, dtype=np.float32)

    for mask, p2p_map in zip(masks_list, pixel_to_point_maps_list):
        view_votes = np.zeros(num_points, dtype=np.float32)
        view_counts = np.zeros(num_points, dtype=np.float32)
        
        if num_point2pixel == 1:
            # Single point per pixel
            valid_pixels = p2p_map != -1
            points = p2p_map[valid_pixels]
            values = mask[valid_pixels]
            
            np.add.at(view_votes, points, values)
            np.add.at(view_counts, points, 1)
        else:
            # Multiple points per pixel
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if len(p2p_map[i,j]) > 0:  # If points exist for this pixel
                        points = p2p_map[i,j]
                        value = mask[i,j]
                        np.add.at(view_votes, points, value)
                        np.add.at(view_counts, points, 1)
        
        # Normalize per view
        valid_view_points = view_counts > 0
        view_votes[valid_view_points] /= view_counts[valid_view_points]
        
        point_votes += view_votes
        point_counts += (view_counts > 0)
    
    # Average across views
    valid_points = point_counts > 0
    point_cloud_affordance = np.zeros_like(point_votes)
    point_cloud_affordance[valid_points] = point_votes[valid_points] / point_counts[valid_points]
    
    return point_cloud_affordance

def project_points_to_image(pc, camera_params, dynamic_radius, fixed_radius=0.005, image_size=(512, 512), num_point2pixel=None, device='cuda'):
    radius = get_dynamic_radius(pc) if dynamic_radius else fixed_radius
    rasterizer = get_rasterizer(camera_params, radius, image_size, device)
    fragments = rasterizer(pc)
    
    zbuf = fragments.zbuf[0].cpu().numpy()
    idx_img = fragments.idx[0].cpu().numpy()
    
    if num_point2pixel == 1:
        # Create a 2D numpy array to store a single point index for each pixel
        pixel_to_point_map = np.full(image_size, -1, dtype=np.int64)
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                valid_points = idx_img[i, j][zbuf[i, j] > -1]
                if valid_points.size > 0:
                    pixel_to_point_map[i, j] = valid_points[0]
    else:
        # Use the original list of lists for multiple points per pixel
        pixel_to_point_map = [[[] for _ in range(image_size[1])] for _ in range(image_size[0])]
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                valid_points = idx_img[i, j][zbuf[i, j] > -1]
                pixel_to_point_map[i][j] = valid_points.tolist()
        pixel_to_point_map = np.array(pixel_to_point_map, dtype=object)
    
    return pixel_to_point_map

def create_affordance_mask(dense_point_cloud, dense_afford_pc, camera_params, dynamic_radius, fixed_radius=0.005, image_size=(512, 512), num_point2pixel=None):
    pixel_to_point_map = project_points_to_image(dense_point_cloud, camera_params, dynamic_radius, fixed_radius, image_size, num_point2pixel=num_point2pixel)
    
    afford_indices = set(dense_afford_pc.tolist())
    
    mask = np.zeros(image_size, dtype=np.uint8)
    
    if num_point2pixel == 1:
        # Use numpy operations for efficiency when we have a 2D array
        mask[np.isin(pixel_to_point_map, list(afford_indices))] = 255
    else:
        # Use the original nested loop for the list of lists
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                if any(idx in afford_indices for idx in pixel_to_point_map[i][j]):
                    mask[i, j] = 255
    
    return mask, pixel_to_point_map

def create_affordance_heatmap(dense_point_cloud, afford_probs, camera_params, dynamic_radius, fixed_radius=0.005, image_size=(512, 512), num_point2pixel=None):
    pixel_to_point_map = project_points_to_image(dense_point_cloud, camera_params, dynamic_radius, fixed_radius, image_size, num_point2pixel=num_point2pixel)
    
    heatmap = np.zeros(image_size, dtype=np.float32)
    
    if num_point2pixel == 1:
        # For single point per pixel
        valid_pixels = pixel_to_point_map != -1
        heatmap[valid_pixels] = afford_probs[pixel_to_point_map[valid_pixels]]
    else:
        # For multiple points per pixel
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                if pixel_to_point_map[i][j]:
                    # Average the probabilities of all points mapping to this pixel
                    heatmap[i, j] = np.mean(afford_probs[pixel_to_point_map[i][j]])
    
    return heatmap, pixel_to_point_map

##################################### Render point cloud ##########################################


def render_pc_p3d(pc, camera_params, dynamic_radius=False, fixed_radius=0.005, image_size=(1024, 512), device='cuda'):
    
    radius = get_dynamic_radius(pc) if dynamic_radius else fixed_radius
    rasterizer = get_rasterizer(camera_params, radius, image_size, device)
    
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
        # compositor=NormWeightedCompositor(background_color=(1, 1, 1))
    )
    images = renderer(pc)
    image = images[0, ..., :3].cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def normalize_point_cloud(point_cloud):
    # Center the point cloud
    centroid = np.mean(point_cloud, axis=0)
    point_cloud_centered = point_cloud - centroid
    
    # Scale the point cloud to fit within a unit sphere
    max_distance = np.max(np.linalg.norm(point_cloud_centered, axis=1))
    point_cloud_normalized = point_cloud_centered / max_distance
    
    return point_cloud_normalized

def enhance_point_cloud_structure_preserving(points, selected_points_idx=None, target_num_points=None, noise_factor=0.01):
    # 1. Build KD-Tree for efficient nearest neighbor queries
    tree = cKDTree(points)

    # 2. Calculate the average distance between points
    distances, _ = tree.query(points, k=2)
    avg_distance = np.mean(distances[:, 1])

    # 3. Generate new points and new selected points
    if selected_points_idx is not None:
        selected_points = points[selected_points_idx]
    else:
        selected_points = None

    needed_new_points = target_num_points - len(points)
    needed_new_selected_points = target_num_points - len(selected_points_idx) if selected_points_idx is not None else 0

    # Vectorized creation of new points
    base_indices = np.random.randint(len(points), size=needed_new_points)
    base_points = points[base_indices]

    offsets = np.random.randn(needed_new_points, 3)
    offsets /= np.linalg.norm(offsets, axis=1)[:, np.newaxis]
    offsets *= (avg_distance * np.random.rand(needed_new_points))[:, np.newaxis]

    new_points = base_points + offsets

    # Add small noise
    noise = np.random.normal(0, noise_factor * avg_distance, new_points.shape)
    new_points += noise

    new_selected_points_idx = []
    if selected_points is not None:
        dists = np.linalg.norm(new_points[:, np.newaxis, :] - selected_points[np.newaxis, :, :], axis=2)
        is_close = np.any(dists < 0.5 * avg_distance, axis=1)
        new_selected_points_idx = np.where(is_close)[0] + len(points)

    # 4. Combine original and new points
    enhanced_points = np.vstack((points, new_points))

    # 5. If we have more points than target, randomly sample
    if target_num_points is not None and len(enhanced_points) > target_num_points:
        indices = np.random.choice(len(enhanced_points), target_num_points, replace=False)
        enhanced_points = enhanced_points[indices]
        enhanced_selected_points_idx = [i for i in new_selected_points_idx if i in indices]
        if selected_points_idx is not None:
            enhanced_selected_points_idx.extend([i for i in selected_points_idx if i in indices])
    else:
        enhanced_selected_points_idx = list(new_selected_points_idx)
        if selected_points_idx is not None:
            enhanced_selected_points_idx.extend(selected_points_idx)

    if selected_points_idx is not None:
        return enhanced_points, np.array(enhanced_selected_points_idx)
    else:
        return enhanced_points


def smooth_mask(mask, kernel_size=5):
    # Ensure the mask is in the correct format
    mask = mask.astype(np.uint8)

    # 1. Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 2. Apply dilation followed by erosion (Closing)
    smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return smoothed_mask

def get_pc_color_by_depth(dense_points, device='cuda'):
    z_normalized = (dense_points[:, 2] - dense_points[:, 2].min()) / (dense_points[:, 2].max() - dense_points[:, 2].min())
    z_normalized = torch.Tensor(z_normalized).to(device)
    pc_rgb_color = torch.zeros(dense_points.shape[0], 3, device=device)
    pc_rgb_color[:, 0] = z_normalized      # Red channel
    pc_rgb_color[:, 2] = 1 - z_normalized  # Blue channel
    return pc_rgb_color

def get_pc_color_by_position(dense_points, device='cuda'):
    dense_points = torch.from_numpy(dense_points).float().to(device)
    min_coords, _ = torch.min(dense_points, dim=0)
    max_coords, _ = torch.max(dense_points, dim=0)
    normalized_points = (dense_points - min_coords) / (max_coords - min_coords)
    
    pc_rgb_color = normalized_points * 0.8 + 0.1 # Adjust range to [0.1, 0.9] instead of [0, 1.0]

    return pc_rgb_color