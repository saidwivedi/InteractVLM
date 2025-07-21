import argparse
import os
import sys
import glob
import shutil
import joblib as jl

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.InteractVLM import InteractVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.utils import convert_contacts
from utils.demo_utils import (
    generate_sam_inp_objs,
    process_smplx_mesh_with_contacts,
    process_object_mesh_with_contacts,
)
from datasets.base_contact_dataset import normalize_cam_params
from preprocess_data.constants import HUMAN_VIEW_DICT, OBJS_VIEW_DICT, SMPL_TO_SMPLX_MAPPING


def parse_args(args):
    parser = argparse.ArgumentParser(description="InteractVLM chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--contact_type", default="hcontact", type=str, 
                        help="Type of contact prediction: 'hcontact' for 3D human contact, 'h2dcontact' for 2D human contact, 'ocontact'/'oafford' for object contact")
    parser.add_argument("--img_folder", default="", type=str)
    parser.add_argument("--input_mode", default="folder", type=str, choices=["folder", "file"],
                        help="Input mode: 'folder' for folder-based samples, 'file' for file-based samples (human contact only)")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    img_folder = args.img_folder

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    smpl_to_smlpx_mapping = jl.load(SMPL_TO_SMPLX_MAPPING)['matrix']
    smpl_to_smlpx_mapping = torch.tensor(smpl_to_smlpx_mapping).float().cuda()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    
    kwargs.update({"train_from_LISA": False})
    kwargs.update({"train_from_LLAVA": False})

    model = InteractVLMForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    camera_params_humans, camera_params_objects = None, None

    llava_image_paths, sam_image_paths, overlay_sam_paths, lift2d_dict_paths = [], [], [], []
    
    # Handle different input modes
    if args.input_mode == "folder":
        # Folder-based: each sample in its own folder
        for img_sample_folder in sorted(os.listdir(img_folder)):
            folder_path = os.path.join(img_folder, img_sample_folder)
            if os.path.isdir(folder_path):
                llava_image_paths.append(os.path.join(folder_path, f'{img_sample_folder}.jpg'))
    else:
        # File-based: all samples as files in a single folder
        for img_file in sorted(os.listdir(img_folder)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and '__' in img_file:
                llava_image_paths.append(os.path.join(img_folder, img_file))
    
    # Create output directory for file-based mode
    if args.input_mode == "file":
        global_output_dir = os.path.join(img_folder, 'contact_output')
        os.makedirs(global_output_dir, exist_ok=True)


    ######################################### Prediciting Object Contact #########################################
    if 'oafford' in args.contact_type or 'ocontact' in args.contact_type:
        print(f'-------> Predicting object contact')
        
        # Object contact always uses folder-based structure
        if args.input_mode == "file":
            print("Warning: Object contact requires folder-based structure. Switching to folder mode.")
            args.input_mode = "folder"
            # Re-process image paths for folder mode
            llava_image_paths = []
            for img_sample_folder in sorted(os.listdir(img_folder)):
                folder_path = os.path.join(img_folder, img_sample_folder)
                if os.path.isdir(folder_path):
                    llava_image_paths.append(os.path.join(folder_path, f'{img_sample_folder}.jpg'))
        
        camera_params_objects = OBJS_VIEW_DICT[model.config.oC_sam_view_type]['cam_params']
        view_names = list(camera_params_objects.keys())
        cam_params = [normalize_cam_params(camera_params_objects[view]) for view in view_names]
        cam_params = torch.stack(cam_params).unsqueeze(0).cuda()
        BASE_PROMPT = ['"What type of affordance does the human-object interaction suggest? Then, segment the area on the {class_name} where the human is making contact.",']
        mask_color = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)

        prompts = []
        for base_prompt in BASE_PROMPT:
            for llava_image_path in llava_image_paths:
                object_name = llava_image_path.split('/')[-1].split('__')[0].lower()
                prompts.append(base_prompt.format(class_name=object_name))
                
                # Folder-based: sam_inp_objs in same folder as image
                sam_base_folder = os.path.dirname(llava_image_path) + '/sam_inp_objs'
                
                # Check if sam_inp_objs folder exists, if not generate it  
                if not os.path.exists(sam_base_folder):
                    obj_mesh_path = os.path.join(os.path.dirname(llava_image_path), 'object_mesh.obj')
                    if os.path.exists(obj_mesh_path):
                        print(f'sam_inp_objs not found, generating for {os.path.dirname(llava_image_path)}')
                        generate_sam_inp_objs(obj_mesh_path)
                    else:
                        print(f'Warning: object_mesh.obj not found at {obj_mesh_path}, cannot generate sam_inp_objs')
                        continue
                
                tmp_sam_paths, tmp_overlay_sam_paths = [], []
                for view in view_names:
                    tmp_sam_paths.append(f'{sam_base_folder}/obj_render_color_{view}.png')
                    tmp_overlay_sam_paths.append(f'{sam_base_folder}/obj_render_grey_{view}.png')
                sam_image_paths.append(tmp_sam_paths)
                overlay_sam_paths.append(tmp_overlay_sam_paths)
                lift2d_dict_paths.append(f'{sam_base_folder}/lift2d_dict.pkl')

        assert len(prompts) == len(llava_image_paths) == len(sam_image_paths) == len(lift2d_dict_paths), \
            "Number of prompts, llava images and sam images must be same"

    ######################################### Prediciting 2D Human Contact #########################################
    elif 'h2dcontact' in args.contact_type:
        print(f'-------> Predicting 2D human contact')
        cam_params = torch.rand(1, 5).cuda()  # Dummy camera parameters for 2D
        BASE_PROMPT = ["Segment the area on the human's body that is in direct contact with the {object} in this image."]
        mask_color_cyan = (np.array([0.0, 1.0, 1.0]) * 255).astype(np.uint8)  # Cyan
        mask_color_red = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)

        prompts = []
        for base_prompt in BASE_PROMPT:
            for llava_image_path in llava_image_paths:
                object_name = llava_image_path.split('/')[-1].split('__')[0].lower()
                prompts.append(base_prompt.format(object=object_name))
        
        llava_image_paths = llava_image_paths * len(BASE_PROMPT)
        sam_image_paths = [None] * len(llava_image_paths)  # Not used for 2D
        overlay_sam_paths = [None] * len(llava_image_paths)  # Not used for 2D
        lift2d_dict_paths = [None] * len(llava_image_paths)

        assert len(prompts) == len(llava_image_paths), \
            "Number of prompts and llava images must be same"

    ######################################### Prediciting Human Contact #########################################
    elif 'hcontact' in args.contact_type:
        print(f'-------> Predicting human contact')
        camera_params_humans = HUMAN_VIEW_DICT[model.config.hC_sam_view_type]['cam_params']
        view_names = list(camera_params_humans.keys())
        cam_params = [normalize_cam_params(camera_params_humans[view]) for view in view_names]
        cam_params = torch.stack(cam_params).unsqueeze(0).cuda()
        base_path = './data/hcontact_vitruvian/'
        sam_image_paths = [[f'{base_path}/body_render_norm_{view}.png' for view in view_names]]
        overlay_sam_paths = [[f'{base_path}/smplh_body_render_blue_{view}.png' for view in view_names]]
        BASE_PROMPT = ['Which body parts are in contact with the {object}? Segment these contact areas.']
        mask_color = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)

        prompts = []
        for base_prompt in BASE_PROMPT:
            for llava_image_path in llava_image_paths:
                object_name = llava_image_path.split('/')[-1].split('__')[0].lower()
                prompts.append(base_prompt.format(object=object_name))
        
        llava_image_paths = llava_image_paths * len(BASE_PROMPT)
        sam_image_paths = sam_image_paths * len(llava_image_paths)
        overlay_sam_paths = overlay_sam_paths * len(llava_image_paths)
        lift2d_dict_paths = [None] * len(llava_image_paths)

        assert len(prompts) == len(llava_image_paths) == len(sam_image_paths), \
            "Number of prompts, llava images and sam images must be same"

    for prompt, llava_image_path, sam_image_path, overlay_sam_path, lift2d_dict_path in \
                            zip(prompts, llava_image_paths, sam_image_paths, overlay_sam_paths, lift2d_dict_paths): 

        # Set output path based on input mode and contact type
        if args.input_mode == "folder":
            args.vis_save_path = os.path.dirname(llava_image_path) + '/contact_output'
        elif 'h2dcontact' in args.contact_type:
            # For 2D contact, create output directory structure similar to original demo
            fname_base = llava_image_path.split("/")[-1].split(".")[0]
            args.vis_save_path = f'output/contact2d_output/{fname_base}'
        else:
            args.vis_save_path = global_output_dir
        os.makedirs(args.vis_save_path, exist_ok=True)

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        if not (os.path.exists(llava_image_path)):
            print("File not found in {}".format(llava_image_path))
            continue

        image_llava_np = cv2.imread(llava_image_path)
        image_llava_np = cv2.cvtColor(image_llava_np, cv2.COLOR_BGR2RGB)

        image_clip = (
            clip_image_processor.preprocess(image_llava_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        # Handle 2D contact case differently
        if 'h2dcontact' in args.contact_type:
            # For 2D contact, use the original image as SAM input
            orig_size_list = [image_llava_np.shape[:2]]
            sam_img = transform.apply_image(image_llava_np)
            resize_list = [sam_img.shape[:2]]
            sam_img = torch.from_numpy(sam_img).permute(2, 0, 1).contiguous()
            sam_multiview = preprocess(sam_img).unsqueeze(0).unsqueeze(0).cuda()
            
        else:
            # Original 3D contact processing
            valid_mask_region = []
            sam_image = [Image.open(sam_img) for sam_img in sam_image_path]
            sam_multiview = np.stack([np.asarray(sam_img) for sam_img in sam_image], axis=0)
            valid_masks_region = [(sam_img.sum(axis=-1) < 255*3).astype(np.uint8) for sam_img in sam_multiview]
            sam_multiview = [transform.apply_image(sam_img) for sam_img in sam_multiview]
            resize_list = [sam_multiview[0].shape[:2]]
            sam_multiview = torch.stack([preprocess(torch.from_numpy(sam_img).permute(2, 0, 1).contiguous()) for sam_img in sam_multiview])
            sam_multiview = sam_multiview.unsqueeze(0).cuda()

        if args.precision == "bf16":
            sam_multiview = sam_multiview.bfloat16()
            cam_params = cam_params.bfloat16()
        elif args.precision == "fp16":
            sam_multiview = sam_multiview.half()
            cam_params = cam_params
        else:
            sam_multiview = sam_multiview.float()
            cam_params = cam_params

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output = model.evaluate(
            image_clip,
            sam_multiview,
            input_ids,
            cam_params,
            resize_list,
            original_size_list=resize_list if 'h2dcontact' not in args.contact_type else orig_size_list,
            lift2d_dict_path=lift2d_dict_path,
            contact_type=args.contact_type,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids, pred_masks = output["output_ids"], output["pred_masks"]

        # Handle 2D contact output
        if 'h2dcontact' in args.contact_type:
            # Decode the output text
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            print(f'\n---> {llava_image_path.split("/")[-1]}: {text_output}')
            
            pred_masks = pred_masks[0][0]

            if pred_masks is not None:
                binary_mask = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)

                alpha = 0.6
                mask_color_cyan = (np.array([0.0, 1.0, 1.0]) * 255).astype(np.uint8)  # Cyan
                mask_color_red = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)

                # Save original image
                output_image_path = os.path.join(args.vis_save_path, 'image.png')
                cv2.imwrite(output_image_path, cv2.cvtColor(image_llava_np, cv2.COLOR_RGB2BGR))
                
                # Red overlay
                overlay_image_red = image_llava_np.copy()
                overlay_image_red[binary_mask == 1] = mask_color_red
                final_overlay_red = cv2.addWeighted(
                    image_llava_np, 1 - alpha, overlay_image_red, alpha, 0)
                fname = llava_image_path.split("/")[-1].split(".")[0]
                model_name = args.version.split("/")[-1]
                output_image_path = os.path.join(args.vis_save_path, f'{model_name}_{fname}_red.png')
                cv2.imwrite(output_image_path, cv2.cvtColor(final_overlay_red, cv2.COLOR_RGB2BGR))
                print(f"Saved red overlay image to {output_image_path}")

                # Cyan overlay
                overlay_image_cyan = image_llava_np.copy()
                overlay_image_cyan[binary_mask == 1] = mask_color_cyan
                final_overlay_cyan = cv2.addWeighted(
                    image_llava_np, 1 - alpha, overlay_image_cyan, alpha, 0)
                output_image_path = os.path.join(args.vis_save_path, f'{model_name}_{fname}_cyan.png')
                cv2.imwrite(output_image_path, cv2.cvtColor(final_overlay_cyan, cv2.COLOR_RGB2BGR))
                print(f"Saved cyan overlay image to {output_image_path}")
        else:
            # Original 3D contact processing
            # Save 3D contact vertices
            pred_contact_3d = output["pred_contact_3d"].detach()
            print(f'---> Num of non-zero contact vertices: {pred_contact_3d[pred_contact_3d != 0].shape[0]}')

            # Determine save path for vertices based on input mode
            if args.input_mode == "folder":
                vertices_save_dir = os.path.dirname(llava_image_path)
            else:
                vertices_save_dir = args.vis_save_path
            
            fname_base = llava_image_path.split("/")[-1].split(".")[0]

            if args.contact_type == 'hcontact':
                pred_contact_3d_smplx = convert_contacts(pred_contact_3d, smpl_to_smlpx_mapping)
                np.savez(f'{vertices_save_dir}/{fname_base}_hcontact_vertices.npz', \
                        pred_contact_3d_smplh=pred_contact_3d.cpu(), pred_contact_3d_smplx=pred_contact_3d_smplx.cpu())
                
                # Process SMPLX mesh with contact vertices
                output_smplx_path = os.path.join(args.vis_save_path, f'{fname_base}_smplx_body_with_hcontacts.obj')
                process_smplx_mesh_with_contacts(
                    pred_contact_3d_smplx, 
                    output_smplx_path,
                    contact_threshold=0.3,
                    gender='neutral'
                )
            else:
                np.savez(f'{vertices_save_dir}/{fname_base}_oafford_vertices.npz', pred_contact_3d=pred_contact_3d.cpu())
                
                # Process object mesh with contact vertices for ocontact/oafford
                # Object contact always uses folder-based structure
                obj_mesh_path = os.path.join(os.path.dirname(llava_image_path), 'object_mesh.obj')
                    
                if os.path.exists(obj_mesh_path):
                    output_obj_path = os.path.join(args.vis_save_path, f'{fname_base}_object_mesh_with_contacts_{args.contact_type}.obj')
                    process_object_mesh_with_contacts(
                        obj_mesh_path, 
                        pred_contact_3d[0], 
                        output_obj_path,
                        contact_threshold=0.5
                    )
                else:
                    print(f'Warning: object mesh not found at {obj_mesh_path}')
            
            # Decode the output text
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            print(f'\n---> {llava_image_path.split("/")[-1]}: {text_output}')
            
            pred_masks = pred_masks[0]

            overlay_images = []
            if pred_masks.shape[0] == 0:
                continue
            for i, pred_mask in enumerate(pred_masks):
                pred_mask = pred_mask.detach().cpu().numpy()
                pred_mask = pred_mask > 0.3 if args.contact_type == 'hcontact' else pred_mask > 0.5
                
                overlay_sam_view_path = overlay_sam_path[i]
                overlay_sam = cv2.imread(overlay_sam_view_path)
                overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_BGR2RGB)

                valid_mask_region = valid_masks_region[i]
                pred_mask = np.logical_and(pred_mask, valid_mask_region)
                
                # Expand pred_mask to match the RGB channels
                pred_mask_3d = np.stack([pred_mask] * 3, axis=2)
                
                # Apply the mask and ensure result is uint8
                overlay_sam = np.where(pred_mask_3d, 
                                    overlay_sam * 0.5 + mask_color * 0.5,
                                    overlay_sam)
                overlay_sam = np.clip(overlay_sam, 0, 255).astype(np.uint8)
                
                overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_RGB2BGR)
                
                # Store the overlay image
                overlay_images.append(overlay_sam)
                
            # Create 2x2 grid
            h, w = overlay_images[0].shape[:2]
            grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)

            # Place images in grid
            grid[:h, :w] = overlay_images[0]  # top-left
            grid[:h, w:] = overlay_images[1]  # top-right
            grid[h:, :w] = overlay_images[2]  # bottom-left
            grid[h:, w:] = overlay_images[3]  # bottom-right

            # Save concatenated image
            fname = llava_image_path.split("/")[-1].split(".")[0]
            model_name = args.version.split("/")[-1]
            
            if args.input_mode == "file" and "hcontact" in args.contact_type:
                # For file mode with hcontact, create combined image with input and grid
                input_image = cv2.imread(llava_image_path)
                input_h, input_w = input_image.shape[:2]
                grid_h, grid_w = grid.shape[:2]
                
                # Resize input image to match grid height
                if input_h != grid_h:
                    aspect_ratio = input_w / input_h
                    new_width = int(grid_h * aspect_ratio)
                    input_image_resized = cv2.resize(input_image, (new_width, grid_h))
                else:
                    input_image_resized = input_image
                    new_width = input_w
                
                # Create combined image: input on left, grid on right
                combined_width = new_width + grid_w
                combined_image = np.zeros((grid_h, combined_width, 3), dtype=np.uint8)
                combined_image[:, :new_width] = input_image_resized
                combined_image[:, new_width:] = grid
                
                combined_save_path = f'{args.vis_save_path}/{model_name}_{fname}_{args.contact_type}_combined.jpg'
                cv2.imwrite(combined_save_path, combined_image)
                print("Combined image saved at: {}".format(combined_save_path))
            else:
                # Original behavior for folder mode or object contact
                concat_save_path = f'{args.vis_save_path}/{model_name}_{fname}_{args.contact_type}_concat.jpg'
                cv2.imwrite(concat_save_path, grid)
                print("Concatenated image saved at: {}".format(concat_save_path))
                shutil.copy(llava_image_path, args.vis_save_path)

if __name__ == "__main__":
    main(sys.argv[1:])
