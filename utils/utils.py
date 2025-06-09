from enum import Enum

import numpy as np
import torch
import torchvision.transforms as T
import torch.distributed as dist

import os
import shutil
import wandb

SAM_MEAN_PIXEL = [123.675, 116.28, 103.53]
SAM_STD_PIXEL = [58.395, 57.12, 57.375]
LLAVA_MEAN_PIXEL = [0.48145466, 0.4578275, 0.40821073]
LLAVA_STD_PIXEL = [0.26862954, 0.26130258, 0.27577711]

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IGNORE_LABEL = -1
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please output segmentation mask.",
]

HCONTACT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the area on the human's body that is in direct contact with the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Identify and mask the part of the human that is touching or interacting with the {class_name} in this scene.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Show the contact points on the human where they are physically connected to or interacting with {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please provide a segmentation mask of the human's body parts that are in contact with {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Highlight the areas on the human where there is physical interaction or contact with {class_name}.",
]

HCONTACT_PARTS_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Which body parts are in contact with the {class_name}? Segment these contact areas.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Name and segment the specific body parts making contact with the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Looking at the {class_name}, what parts of the human body are touching it? Show these contact regions.",
    DEFAULT_IMAGE_TOKEN + "\n" + "For the {class_name}, list and mask the human body parts that are in contact.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Regarding the {class_name}, identify which body parts are touching it and highlight these contact areas."
]

OAFFORD_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the area on the {class_name} where the human is making direct contact in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Identify and mask the part of the {class_name} that the human is touching or interacting with in this scene.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Show the contact points on the {class_name} where the human is physically connected to or interacting with it.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please provide a segmentation mask of the parts of the {class_name} that are in contact with the human.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Highlight the areas on the {class_name} where there is physical interaction or contact with the human.",
]

# TODO: Check if one needs to add more questions for OCONTACT
OCONTACT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the area on the {class_name} where the human is making direct contact in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Identify and mask the part of the {class_name} that the human is touching or interacting with in this scene.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Show the contact points on the {class_name} where the human is physically connected to or interacting with it.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please provide a segmentation mask of the parts of the {class_name} that are in contact with the human.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Highlight the areas on the {class_name} where there is physical interaction or contact with the human.",
]

OAFFORD_AFFORD_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "What type of affordance does the human-object interaction suggest? Then, segment the area on the {class_name} where the human is making contact.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Describe the affordance provided by the interaction, and identify the part of the {class_name} that the human is touching or interacting with in this scene.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Explain the affordance type shown by the contact points on the {class_name} where the human is physically connected. Then show the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Specify the affordance implied by the human's contact with the {class_name}, then provide a segmentation mask of the contact area.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Describe the affordance associated with the physical interaction on the {class_name}, and highlight the contact areas with a segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

HCONTACT_ANSWER_LIST = [
    "It is [HTOKEN].",
    "Sure, the human contact region is [HTOKEN].",
    "Sure, the contact points on human is [HTOKEN].",
    "Sure, the contact mask is [HTOKEN].",
    "[HTOKEN].",
]

HCONTACT_PARTS_ANSWER_LIST = [
    "The contacting body parts are {body_parts}, and the contact region is [HTOKEN].",
    "The involved body parts are {body_parts}, with the contact mask at [HTOKEN].",
    "Contact occurs at {body_parts}, with the contact points shown at [HTOKEN].",
    "The body parts in contact are {body_parts}, with contact mask at [HTOKEN].",
    "Body parts: {body_parts}, contact mask: [HTOKEN].",
]

OAFFORD_ANSWER_LIST = [
    "It is [OTOKEN].",
    "Sure, the object contact region is [OTOKEN].",
    "Sure, the contact points on object is [OTOKEN].",
    "Sure, the contact mask is [OTOKEN].",
    "[OTOKEN].",
]

OCONTACT_ANSWER_LIST = [
    "It is [OTOKEN].",
    "Sure, the object contact region is [OTOKEN].",
    "Sure, the contact points on object is [OTOKEN].",
    "Sure, the contact mask is [OTOKEN].",
    "[OTOKEN].",
]

OAFFORD_AFFORD_ANSWER_LIST = [
    "The affordance type is {affordance}, and the contact region is [OTOKEN].",
    "This interaction suggests an affordance of {affordance}, and the object contact region is [OTOKEN].",
    "The contact points indicate an affordance of {affordance}, with the mask at [OTOKEN].",
    "This shows an affordance type of {affordance}, with contact at [OTOKEN].",
    "Affordance: {affordance}, contact mask: [OTOKEN].",
]

OAFFORD_AFFORD_OBJ_ANSWER_LIST = [
    "The affordance type is {affordance} with {class_name}, and the contact region is [OTOKEN].",
    "This interaction suggests an affordance of {affordance} with {class_name}, and the object contact region is [OTOKEN].",
    "The contact points indicate an affordance of {affordance} with {class_name}, with the mask at [OTOKEN].",
    "This shows an affordance type of {affordance} with {class_name}, with contact at [OTOKEN].",
    "Affordance: {affordance} with {class_name}, contact mask: [OTOKEN].",
]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Convert val to a tensor if it's not already one
        if not torch.is_tensor(val):
            val = torch.tensor(val)

        # Check if val contains any NaNs
        if not torch.isnan(val).any() or not torch.isinf(val).any():  # Ensure no NaNs in val
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            print(f"Skipping update due to NaN in {self.name}")

    def all_reduce(self):
        if not torch.is_tensor(self.sum):
            self.sum = torch.tensor(self.sum)

        # Get the current device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
        
        # Move sum to the correct device and flatten if it's multi-dimensional
        flat_sum = self.sum.to(device).reshape(-1)
        total = torch.cat([flat_sum, torch.tensor([self.count], device=device)])

        # Perform all_reduce operation
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        
        # Reshape sum back to original shape
        sum_size = self.sum.numel()
        self.sum = total[:sum_size].reshape(self.sum.shape).cpu()
        self.count = total[-1].cpu().item()
        
        self.avg = self.sum / (self.count + 1e-5)


    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def debug_torch_tensor(tensor, name, verbose=False):

    print(f"############################################")
    print(f"{name}: Torch: shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
    if tensor.numel() == 0:
        print("\n"); return
    
    torch.set_printoptions(profile="full")
    if tensor.dtype == torch.bool:
        print(f"{name}: Torch: True count: {tensor.sum().item()}, False count: {(~tensor).sum().item()}\n")
        return  # No further operations needed for boolean tensors

    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"{name}: Torch: contains NaN or Inf values!")
        if verbose:
            non_nan_inf = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
            if non_nan_inf.numel() > 0:
                print(f"{name}: Torch: non-NaN/Inf stats: min={non_nan_inf.min().item():.4f}, max={non_nan_inf.max().item():.4f}")
                if torch.is_floating_point(non_nan_inf):
                    print(f"{name}: Torch: non-NaN/Inf mean: {non_nan_inf.mean().item():.4f}\n")
            else:
                print(f"{name} contains only NaN or Inf values\n")
    else:
        print(f"{name}: Torch: stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        
        if torch.is_floating_point(tensor):
            print(f"{name}: Torch: mean: {tensor.mean().item():.4f}\n")
        else:
            # Additional information for non-floating point tensors
            unique_vals = torch.unique(tensor)
            print(f"{name}: Torch: unique values: {unique_vals.numel()} ({unique_vals})")  # Print first 5 unique values
            print(f"{name}: Torch: sum: {tensor.sum().item()}, median: {tensor.median().item()}\n")

    torch.set_printoptions(profile="default")

def debug_numpy_array(array, name, verbose=False):

    print(f"############################################")
    print(f"{name}: Numpy: shape: {array.shape}, dtype: {array.dtype}")
    if array.size == 0:
        print("\n"); return
    
    default_numpy_print_options = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)

    if array.dtype == np.bool_:
        print(f"{name}: Numpy: True count: {array.sum()}, False count: {(~array).sum()}\n")
        return  # No further operations needed for boolean arrays

    if np.isnan(array).any() or np.isinf(array).any():
        print(f"{name}: Numpy: contains NaN or Inf values!")
        if verbose:
            non_nan_inf = array[~np.isnan(array) & ~np.isinf(array)]
            if non_nan_inf.size > 0:
                print(f"{name}: Numpy: non-NaN/Inf stats: min={non_nan_inf.min():.4f}, max={non_nan_inf.max():.4f}")
                if np.issubdtype(non_nan_inf.dtype, np.floating):
                    print(f"{name}: Numpy: non-NaN/Inf mean: {non_nan_inf.mean():.4f}\n")
            else:
                print(f"{name}: Numpy: contains only NaN or Inf values\n")
    else:
        print(f"{name}: Numpy: stats: min={array.min():.4f}, max={array.max():.4f}")
        if np.issubdtype(array.dtype, np.floating):
            print(f"{name}: Numpy: mean: {array.mean():.4f}\n")
        else:
            # Additional information for non-floating point arrays
            unique_vals = np.unique(array)
            print(f"{name}: Numpy: unique values: {len(unique_vals)} ({unique_vals})")  # Print first 5 unique values
            print(f"{name}: Numpy: sum: {array.sum()}, median: {np.median(array)}\n")
    
    np.set_printoptions(**default_numpy_print_options)

def debug_list_of_tensors(tensor_list, name, verbose=False):
    print(f"############################################")
    print(f'{name} is a list with {len(tensor_list)} elements, showing first element:')
    if len(tensor_list) > 0:
        if isinstance(tensor_list[0], torch.Tensor):
            debug_torch_tensor(tensor_list[0], f'{name}[0]', verbose)
        elif isinstance(tensor_list[0], np.ndarray):
            debug_numpy_array(tensor_list[0], f'{name}[0]', verbose)
        else:
            print(f"{name} List: {tensor_list}\n")

def debug_tensor(tensor, name, verbose=False):
    if isinstance(tensor, torch.Tensor):
        debug_torch_tensor(tensor, name, verbose)
    elif isinstance(tensor, np.ndarray):
        debug_numpy_array(tensor, name, verbose)
    elif isinstance(tensor, list):
        debug_list_of_tensors(tensor, name, verbose)
    else:
        tensor_type = type(tensor).__name__
        print(f"\n{name} is {tensor_type}\n")

def add_new_tokens(tokenizer, args):

    def add_token(tokenizer, token):
        tokenizer.add_tokens(token)
        token_idx = tokenizer(token, add_special_tokens=False)["input_ids"][0]
        print(f"\nAdded token: {token}, token_idx: {token_idx}\n")
        return tokenizer, token_idx

    ## Three types of tokens:
    ## 1. Gen - For general segmentation
    ## 2. Gen-Int - For segmentation with joint token for human and object
    ## 3. Gen-Hu-Obj - For segmentation with separate tokens for human and object

    args.seg_token_idx, args.hseg_token_idx, args.oseg_token_idx = None, None, None
    base_token_type = args.token_type.replace('-DifDe', '')
    if base_token_type == 'Gen':
        tokenizer, args.seg_token_idx = add_token(tokenizer, "[SEG]")
    elif base_token_type == 'Gen-Int':
        tokenizer, args.seg_token_idx = add_token(tokenizer, "[SEG]")
        tokenizer, args.hseg_token_idx = add_token(tokenizer, "[ISEG]")
        tokenizer, args.oseg_token_idx = add_token(tokenizer, "[ISEG]")
    elif base_token_type == 'Gen-Hu-Obj':
        tokenizer, args.seg_token_idx = add_token(tokenizer, "[SEG]")
        tokenizer, args.hseg_token_idx = add_token(tokenizer, "[HSEG]")
        tokenizer, args.oseg_token_idx = add_token(tokenizer, "[OSEG]")
    else:
        print(f'\n-----> Warning: Token type {args.token_type} not recognized\n')
    return tokenizer, args


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

def resize_images_for_tb(images, target_size=(128, 128)):
    transform = T.Resize(target_size, antialias=True)
    resized_images = []
    for image in images:
        image[image == IGNORE_LABEL] = 0
        image = transform(image).cpu().clip(0, 1)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        resized_images.append(image)
    return resized_images

def get_dtype(precision):
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32
    
def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(image.device)
    std = torch.tensor(std).view(-1, 1, 1).to(image.device)
    image = (image * std) + mean    
    return image

def copy_code(output_dir):
    
    dest_dir = f'{output_dir}/code'
    os.makedirs(dest_dir, exist_ok=True)

    include_dirs = ['.', 'model', 'preprocess_data', 'utils']  # Add the directories you want to include

    # Function to copy .py files from a specific directory
    def copy_py_files(src_dir, dest_base):
        for file_name in os.listdir(src_dir):
            if file_name.endswith('.py') or file_name.endswith('.ipynb') or file_name.endswith('.sh'):
                full_file_name = os.path.join(src_dir, file_name)
                if os.path.isfile(full_file_name):
                    rel_path = os.path.relpath(src_dir, '.')
                    dest_path = os.path.join(dest_base, rel_path)
                    os.makedirs(dest_path, exist_ok=True)
                    shutil.copy(full_file_name, dest_path)

    # Copy .py files from each directory in include_dirs
    for directory in include_dirs:
        print(f'Copying code from {directory} to {dest_dir}')
        copy_py_files(directory, dest_dir)

    print(f'\n Code copied to {dest_dir}')


def convert_contacts(contact_labels, mapping):
    """
    Converts the contact labels from SMPL to SMPL-X format and vice-versa.

    Args:
        contact_labels: contact labels in SMPL or SMPL-X format
        mapping: mapping from SMPL to SMPL-X vertices or vice-versa

    Returns:
        contact_labels_converted: converted contact labels
    """
    bs = contact_labels.shape[0]
    mapping = mapping[None].expand(bs, -1, -1)
    contact_labels_converted = torch.bmm(mapping, contact_labels[..., None])
    contact_labels_converted = contact_labels_converted.squeeze()
    return contact_labels_converted

def log_images(loggers, idx, tag, input_dict, output_dict, global_step, disp_size=128):
    writer, wandb_logger = loggers
    random_view = np.random.randint(0, input_dict["images"][idx].shape[0]) # select a random view incase of multi-view channels
    llava_img = denormalize_image(input_dict["images_clip"][idx].float(), LLAVA_MEAN_PIXEL, LLAVA_STD_PIXEL)
    sam_img = denormalize_image(input_dict["images"][idx][random_view].float(), SAM_MEAN_PIXEL, SAM_STD_PIXEL)
    sam_img = torch.clamp((sam_img / 255.0), 0.0, 1.0)
    pred_mask = output_dict["pred_masks"][idx].float()
    gt_mask = output_dict["gt_masks"][idx].float()
    try:
        if pred_mask.shape[0] != 0:
            pred_mask = pred_mask[random_view].unsqueeze(0)
            gt_mask = gt_mask[random_view].unsqueeze(0)
            display_images = [llava_img, sam_img, pred_mask, gt_mask]
            if "uncertainty_maps" in output_dict and len(output_dict["uncertainty_maps"]) > idx:
                uncertainty_map = output_dict["uncertainty_maps"][idx].float()
                display_images.append(uncertainty_map)
            display_images = resize_images_for_tb(display_images, (disp_size, disp_size))
            display_images = torch.cat(display_images, dim=2)
            if wandb_logger:
                wandb_logger.log({tag: [wandb.Image(display_images)]})
            if writer:
                writer.add_image(tag, display_images, global_step)
        else:
            print(f"Invalid pred_mask shape: {pred_mask.shape}, cannot log images")
    except Exception as e:
        print(f"error in logging train images: {e}")

def log_metric(logger, metric_dict, step):
    writer, wandb_logger = logger
    if writer:
        for key, value in metric_dict.items():
            writer.add_scalar(key, value, step)             
    if wandb_logger:
        wandb_logger.log(metric_dict)