import copy
import json
import logging
import time
import traceback
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pytorch3d.structures import Pointclouds
from tqdm import tqdm

from .data_io import load_image, load_params
from .icp import ICP, SimilarityTransform
from .optimizer import ObjPose_Opt
from .renderer import HPRenderer, SSRenderer
from .utils import (
    Config,
    EasierDict,
    fix_seeds,
    human_readable_time,
    matrix_to_rot6d,
    rot6d_to_matrix,
)

fix_seeds(42)


def setup_logging(output_dir: Path, log_to_file: bool):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    if log_to_file:
        log_file = output_dir / "fitting.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(stream_handler)


def get_optimizer(name, parameter_list):
    if name == "adam":
        return torch.optim.Adam(parameter_list)
    elif name == "sgd":
        return torch.optim.SGD(parameter_list)
    else:
        raise ValueError(f"Unknown optimizer {name}")


def main(input_path: Path, opt: EasierDict):
    cfg = Config()

    logging.info(f"\nProcessing {input_path}")
    if opt.out_root is None:
        output_dir = input_path / "results"
    else:
        output_dir = Path(opt.out_root) / f"{input_path.parts[-1]}"
    output_dir.mkdir(exist_ok=True, parents=True)
    yaml.dump(opt.to_dict(), open(output_dir / "cfg.yaml", "w"))

    setup_logging(output_dir, opt.log_out == "file")

    logging.info(f"Optimization config: {json.dumps(opt, indent=4)}")
    logging.info(f"\nSaving results to {output_dir}")

    img = load_image(input_path)
    logging.info(f"Loaded {input_path} with shape {img.shape}")

    human_params, object_params, camera_params = load_params(
        input_path.parent / cfg.human_inference_file,
        input_path.parent / cfg.object_mesh_file,
        input_path.parent / cfg.object_detection_file,
    )

    silhouette_renderer = SSRenderer(
        img.shape,
        human_params.faces,
        object_params.faces,
        camera_params,
    )
    phong_renderer = HPRenderer(
        img.shape,
        human_params.faces,
        object_params.faces,
        camera_params,
    )

    # * Initialize the optimizable parameters
    rotation_init = matrix_to_rot6d(torch.eye(3).unsqueeze(0)).cuda()
    translation_init = torch.tensor([0.0, 0.0, 0.0]).cuda()
    scaling_init = torch.tensor([object_params.scale]).cuda()

    start_time = time.time()

    # Shortened
    h_verts = human_params.vertices
    h_norms = human_params.normals
    h_contact_probs = human_params.contact_verts
    h_contact_mask = h_contact_probs > 0.5

    o_verts = object_params.vertices
    o_norms = object_params.normals
    o_contact_probs = object_params.contact_verts
    o_contact_mask = o_contact_probs > 0.3

    init_opt = opt.init
    icp_opt = init_opt.icp

    if init_opt.translation_hum_centroid:
        logging.info("Init with centroid estimate")
        obj_mask = object_params.mask
        obj_mask_indices = torch.nonzero(obj_mask).float()
        # hum_centroid_z = human_params.vertices[:, 2].mean()
        hum_centroid_z = h_verts[h_contact_mask, 2].mean()
        cx = obj_mask_indices[1].mean() - camera_params.principal_point[0]
        cy = obj_mask_indices[0].mean() - camera_params.principal_point[1]
        translation_x = cx * hum_centroid_z / camera_params.focal_length[0]
        translation_y = cy * hum_centroid_z / camera_params.focal_length[1]
        translation_init = torch.tensor(
            [
                translation_x,
                translation_y,
                hum_centroid_z,
            ]
        ).cuda()

    if icp_opt.run:
        logging.info("Running ICP")
        print("o_contact_mask", o_contact_mask.sum())
        if icp_opt.filter_contacts[0]:
            # * Filter out points based on normals direction
            h_contact_normals = F.normalize(-h_norms[h_contact_mask], p=2, dim=-1)
            o_contact_normals = F.normalize(o_norms[o_contact_mask], p=2, dim=-1)
            cosine_threshold = torch.cos(
                torch.deg2rad(
                    torch.tensor(icp_opt.filter_contacts[1], dtype=torch.float32),
                )
            )

            dot_products = torch.mm(o_contact_normals, h_contact_normals.T)

            # Apply the angle threshold to filter out dissimilar pairs
            valid_pairs = dot_products > cosine_threshold

            if len(icp_opt.filter_contacts) == 3:
                cosine_threshold_neg = torch.cos(
                    torch.deg2rad(
                        torch.tensor(icp_opt.filter_contacts[2], dtype=torch.float32),
                    )
                )
                valid_pairs = valid_pairs | (dot_products < cosine_threshold_neg)

            best_matches = valid_pairs.any(dim=1)
            o_contact_mask[o_contact_mask.clone()] = best_matches

            object_params.contact_verts[o_contact_mask] = o_contact_probs[o_contact_mask]
            object_params.contact_verts[~o_contact_mask] = 0.0

        human_contact_pcd = Pointclouds(points=[h_verts[h_contact_mask]])
        object_contact_pcd = Pointclouds(points=[o_verts[o_contact_mask]])

        # * Extract contacts again since they might've changed
        o_contact_probs = object_params.contact_verts
        o_contact_mask = o_contact_probs > 0.3

        icp_solution = ICP(
            obj_contact_pcd=object_contact_pcd,
            obj_contact_normals=o_norms[o_contact_mask, :].unsqueeze(0),
            #
            hum_contact_pcd=human_contact_pcd,
            hum_contact_normals=h_norms[h_contact_mask > 0.5, :].unsqueeze(0),
            #
            max_iterations=icp_opt.max_iter,
            estimate_scale=icp_opt.est_scale,
            init_transform=SimilarityTransform(
                R=rot6d_to_matrix(rotation_init.unsqueeze(0)),
                T=translation_init.unsqueeze(0),
                s=scaling_init,
            ),
        )

        rotation_init = matrix_to_rot6d(icp_solution.RTs.R)
        translation_init = icp_solution.RTs.T.squeeze()

    logging.info(f"Initial guess: {rotation_init=}, {translation_init=}, {scaling_init=}")
    logging.info(f"Optimizing: {opt.vars}")

    # * Initialize the model
    model = ObjPose_Opt(
        rotation_init,
        translation_init,
        scaling_init,
        human_params,
        object_params,
        img,
        silhouette_renderer,
        phong_renderer,
        vars=opt.vars,
        log_dir=output_dir,
    )
    model.cuda()

    init_time = human_readable_time(time.time() - start_time)
    logging.info(f"Initialization took {init_time}")
    loss_weights = copy.deepcopy(opt.loss_weights)

    # optimizer with separate learning rates for each parameter
    parameter_list = [
        {"params": [model.rotation], "lr": 5.0e-2},
        {"params": [model.translation], "lr": 1.0e-2},
    ]
    # if opt.en_scale_loss:
    if "scale" in opt.vars:
        parameter_list.append({"params": [model.scale], "lr": 1.0e-2})

    optimizer = get_optimizer(opt.optim_name, parameter_list)

    if opt.log_video:
        from videoio import VideoWriter

        h, w = img.shape[:2]
        original_width = 2 * w
        new_width = min(1024, original_width)
        new_height = int(h * (new_width / original_width))

        # Round up to the nearest multiple of 16 for better video encoding
        new_width = (new_width + 15) // 16 * 16
        new_height = (new_height + 15) // 16 * 16

        vwriter = VideoWriter(
            output_dir / "video.mp4",
            resolution=(new_width, new_height),
            fps=30,
            preset="ultrafast",
            lossless=False,
        )

    try:
        pbar = tqdm(range(opt.max_iter))
        prev_loss = 1e10
        for i in pbar:
            optimizer.zero_grad()
            loss, pbar_str, output = model(loss_weights)
            loss.backward()
            optimizer.step()

            loss_diff = prev_loss - loss.item()
            pbar_str = f"(prev-curr)x10^4: {loss_diff * 1.0e4:.4f}{pbar_str}"

            pbar.set_description(pbar_str)
            if i % opt.print_every == 0 and opt.log_out == "file":
                logging.info(f"iter={i:04d} {pbar_str}")

            if opt.log_video:
                target_centroid = model.target_mask_centroid.cpu().numpy().astype(int)

                target_mask = model.target_mask.squeeze().cpu().numpy().copy()
                display_target_mask = cv2.cvtColor(target_mask, cv2.COLOR_GRAY2RGB)
                display_target_mask = (
                    cv2.circle(
                        display_target_mask,
                        target_centroid[::-1],
                        12,
                        color=(1.0, 0.0, 0.0),
                        thickness=8,
                    )
                    * 255.0
                )
                hardP_render_overlay = output["hardP_render_overlay"]
                frame = np.hstack((img, hardP_render_overlay)) / 255
                frame = cv2.resize(frame, (new_width, new_height))
                vwriter.write(frame)

            if opt.early_stop and np.abs(loss_diff) < 1e-6:
                logging.info(
                    f"Early stopping at iteration {i} with loss {loss.item():.4f}, diff {loss_diff}"
                )
                break

            prev_loss = loss.item()

            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/final_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{output_dir}/input_image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                f"{output_dir}/overlay_output.jpg",
                cv2.cvtColor(hardP_render_overlay, cv2.COLOR_RGB2BGR),
            )

        end_time = human_readable_time(time.time() - start_time)
        logging.info(f"Total runtime {end_time}" + " with logging" if opt.log_video else "")
    except Exception:
        traceback.print_exc()
        if (output_dir / "video.mp4").exists():
            (output_dir / "video.mp4").unlink()
    finally:
        # * Final result logging
        phong_renderer.save_mesh_as_obj(
            model.human_vertices,
            output["object_vertices"],
            output_dir / "final.obj",
            separate=True,
        )
        if opt.log_video:
            vwriter.close()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--input_path", type=Path, required=True)
    args.add_argument("--cfg", type=Path, required=True)
    args.add_argument("--out_root", type=str, default=None)

    # Add any extra args that will be parsed by OmegaConf
    args, remaining_args = args.parse_known_args()

    conf = OmegaConf.load(args.cfg)

    cli_conf = OmegaConf.from_cli(remaining_args)
    cleaned_cli_conf = OmegaConf.create(
        {k.lstrip("+"): v for k, v in OmegaConf.to_container(cli_conf).items()}
    )

    # Merge configs (CLI overrides base config)
    cfg = OmegaConf.merge(conf, cleaned_cli_conf)

    # Override out_root from ArgumentParser if provided
    if args.out_root is not None:
        cfg.out_root = args.out_root

    opt = EasierDict(OmegaConf.to_container(cfg, resolve=True))

    main(Path(args.input_path), opt)
