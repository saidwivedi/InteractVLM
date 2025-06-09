import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.vis.plotly_vis import plot_scene


def plotly_p3d(meshes, names=None):
    if not isinstance(meshes, list):
        meshes = [meshes]
    if names:
        assert len(names) == len(meshes), 'Number of names should match number of meshes'
    else:
        names = [f'3DModel_{i+1}' for i in range(len(meshes))]
    mesh_dict = dict()
    for i, mesh in enumerate(meshes):
        mesh_dict.update({
            names[i]: {
                "mesh_trace_title": mesh
            }    
        })
    fig = plot_scene(mesh_dict, ncols=len(meshes), )
    fig.show()

def resize_to_square(img_path, output_size, resize_type='crop', background=(0, 0, 0)):
    # Read the image
    img = cv2.imread(img_path)
    
    # Check if image was successfully loaded
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    
    # Get the original dimensions
    height, width = img.shape[:2]
    
    # Calculate the scaling factor to fit within the output size
    if resize_type == 'pad':
        scale = min(output_size / width, output_size / height)
    else:  # crop
        scale = max(output_size / width, output_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    if resize_type == 'pad':
        # Padding logic
        img_padded = np.full((output_size, output_size, 3), background, dtype=np.uint8)
        pad_width = (output_size - new_width) // 2
        pad_height = (output_size - new_height) // 2
        img_padded[pad_height:pad_height+new_height, pad_width:pad_width+new_width] = img_resized
    else:  # crop
        # Crop from the center
        start_x = max(0, (new_width - output_size) // 2)
        start_y = max(0, (new_height - output_size) // 2)
        end_x = min(new_width, start_x + output_size)
        end_y = min(new_height, start_y + output_size)
        
        img_cropped = img_resized[start_y:end_y, start_x:end_x]
        
        # If the cropped image is smaller than the output size, pad it
        if img_cropped.shape[0] < output_size or img_cropped.shape[1] < output_size:
            img_padded = np.full((output_size, output_size, 3), background, dtype=np.uint8)
            pad_width = (output_size - img_cropped.shape[1]) // 2
            pad_height = (output_size - img_cropped.shape[0]) // 2
            img_padded[pad_height:pad_height+img_cropped.shape[0], 
                       pad_width:pad_width+img_cropped.shape[1]] = img_cropped
        else:
            img_padded = img_cropped
    
    # Convert BGR to RGB before returning
    img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    return img_padded

def overlay_text_on_image_cv2(image, text, position, font_size=36, text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=5):
    # Convert colors from RGB to BGR (since OpenCV uses BGR)
    text_color_bgr = tuple(reversed(text_color))
    bg_color_bgr = tuple(reversed(bg_color))
    
    # Specify the font and calculate the thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(font_size // 12, 1)
    
    # Get the size of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size / 36, thickness)
    
    # Calculate the position of the background rectangle
    top_left = (position[0] - padding, position[1] - text_height - padding)
    bottom_right = (position[0] + text_width + padding, position[1] + baseline + padding)
    
    # Draw the background rectangle
    cv2.rectangle(image, top_left, bottom_right, bg_color_bgr, thickness=cv2.FILLED)
    
    # Add the text on top of the rectangle
    cv2.putText(image, text, position, font, font_size / 36, text_color_bgr, thickness, lineType=cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def display_plt_images(image_paths, columns=1):
    rows = len(image_paths) // columns + (len(image_paths) % columns > 0)
    img_height = 512

    # Create the figure and axes
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    background = (0, 0, 0)

    # Iterate over the image paths and axes to display the images
    for i, ax in enumerate(axes.flat):
        if ((i+1) % 3 == 0):
            background = (255, 255, 255)
            img = resize_to_square(image_paths[i], img_height, 'crop', background)
            img[img > 0] = 255
        else:
            background = (0, 0, 0)
            img = resize_to_square(image_paths[i], img_height, 'crop', background)
        ax.imshow(img)
        ax.axis('off')  # Hide axes ticks
    plt.tight_layout()
    plt.show()