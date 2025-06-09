from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import debug_tensor, IGNORE_LABEL

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .components import LLaVASAMFusion, UncertaintyModule, AttentionSplitter
from .components import HumanContact3DPredictor, ObjectPCAfford3DPredictor, ObjectMeshContact3DPredictor
from .components import CamPoseEncoder, ViewIndexCamPoseEncoder, VIv1CamPoseEncoder
from .components import get_initial_weights, check_weight_changes
from .losses import CombinedLoss


class ModifiedSAM(nn.Module):
    def __init__(self, original_sam, use_diff_decoder, use_fusion=False, use_uncertainty=False):
        super().__init__()
        self.image_encoder = original_sam.image_encoder
        self.prompt_encoder = original_sam.prompt_encoder
        self.mask_decoder = original_sam.mask_decoder
        self.postprocess_masks = original_sam.postprocess_masks
        self.use_fusion = use_fusion
        self.use_uncertainty = use_uncertainty
        self.use_diff_decoder = use_diff_decoder
        if self.use_diff_decoder:
            self.human_mask_decoder = original_sam.mask_decoder
            self.object_mask_decoder = original_sam.mask_decoder
        if use_fusion:
            print("Using LLaVASAMFusion for feature fusion")
            self.fusion = LLaVASAMFusion()
        if use_uncertainty:
            print("Using UncertaintyModule for uncertainty estimation")
            self.uncertainty = UncertaintyModule()
    
    def forward(self, image_embeddings, llava_features, sparse_prompt_embeddings, dense_prompt_embeddings, ds_name=None):
        if self.use_fusion:
            fused_embeddings = self.fusion(image_embeddings, llava_features)
        else:
            fused_embeddings = image_embeddings

        if self.use_diff_decoder:
            if 'hcontact' in ds_name:
                mask_decoder = self.human_mask_decoder
            elif 'oafford' or 'ocontact' in ds_name:
                mask_decoder = self.object_mask_decoder
            else:
                mask_decoder = self.mask_decoder
        else:
            mask_decoder = self.mask_decoder

        masks, iou_pred = mask_decoder(
            image_embeddings=fused_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )
        return masks, iou_pred


class InteractVLMMetaModel:
    def __init__(self, config, **kwargs):
        super(InteractVLMMetaModel, self).__init__(config)
        self.config = config
        self.use_fusion = self.config.use_fusion
        self.use_uncertainty = self.config.use_uncertainty
        self.use_diff_decoder = 'DifDe' in self.config.token_type

        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_ivlm_modules(self.config)

    def initialize_ivlm_modules(self, config):
        # SAM
        original_sam = build_sam_vit_h(self.vision_pretrained)
        self.visual_model = ModifiedSAM(original_sam, 
                                        self.use_diff_decoder, 
                                        use_fusion=self.use_fusion, 
                                        use_uncertainty=self.use_uncertainty)
        
        # Freeze parameters of the visual model
        for param in self.visual_model.parameters():
            param.requires_grad = False

        # If training the mask decoder, unfreeze its parameters
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def initialize_separate_decoders(self):
        self.visual_model.human_mask_decoder = copy.deepcopy(self.visual_model.mask_decoder)
        self.visual_model.object_mask_decoder = copy.deepcopy(self.visual_model.mask_decoder)
        self.visual_model.human_mask_decoder.train()
        self.visual_model.object_mask_decoder.train()
        for param in self.visual_model.human_mask_decoder.parameters():
            param.requires_grad = True
        for param in self.visual_model.object_mask_decoder.parameters():
            param.requires_grad = True


class InteractVLMModel(InteractVLMMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(InteractVLMModel, self).__init__(config, **kwargs)
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class InteractVLMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):

        train_from_LISA = kwargs.pop("train_from_LISA", False)
        # TODO: there is might be some bug for training from LLAVA, introduce after mask size commit
        train_from_LLAVA = kwargs.pop("train_from_LLAVA", False)

        if train_from_LISA or train_from_LLAVA:
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            config.use_fusion = kwargs.get("use_feat_fusion", False)
            config.use_uncertainty = kwargs.get('use_uncertainty', False)
            config.img_emb_len = kwargs.get("img_emb_len", 255)
            config.seg_token_idx = kwargs.get("seg_token_idx", None)
            config.hseg_token_idx = kwargs.get("hseg_token_idx", None)
            config.oseg_token_idx = kwargs.get("oseg_token_idx", None)
            config.token_type = kwargs.get("token_type", 'Gen')
            config.hC_sam_view_type = kwargs.get("hC_sam_view_type", None)
            config.oC_sam_view_type = kwargs.get("oC_sam_view_type", None)
            config.hC_loss_weight = kwargs.get("hC_loss_weight", 0.0)
            config.oC_loss_weight = kwargs.get("oC_loss_weight", 0.0)
            config.hC_question_type = kwargs.get("hC_question_type", 'simple')
            config.oC_question_type = kwargs.get("oC_question_type", 'simple')
            config.multiview_channels = kwargs.get("multiview_channels", 4)
            config.multiview_cam_cond = kwargs.get("multiview_cam_cond", False)
            config.cam_encoder_type = kwargs.get("cam_encoder_type", 'simple')

            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.dice_loss_scale =  kwargs.pop("dice_loss_scale", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.bce_loss_alpha = kwargs.pop("bce_loss_alpha", None)
            
        else:
            config.mm_vision_tower = config.vision_tower
            oC_sam_view_type_inf = kwargs.pop("oC_sam_view_type", None)
            if oC_sam_view_type_inf is not None:
                config.oC_sam_view_type = oC_sam_view_type_inf
            config.oC_question_type = kwargs.pop("oC_question_type", 'simple')
            config.hC_question_type = kwargs.pop("hC_question_type", 'simple')

        self.hC_sam_view_type = config.hC_sam_view_type
        self.oC_sam_view_type = config.oC_sam_view_type
        self.hC_loss_weight = config.hC_loss_weight
        self.oC_loss_weight = config.oC_loss_weight
        self.oC_question_type = config.oC_question_type
        self.hC_question_type = config.hC_question_type

        self.seg_token_idx = config.seg_token_idx
        self.hseg_token_idx = config.hseg_token_idx
        self.oseg_token_idx = config.oseg_token_idx
        self.token_type = config.token_type
        self.img_emb_len = config.img_emb_len
        self.multiview_channels = config.multiview_channels
        self.multiview_cam_cond = config.multiview_cam_cond

        self.use_fusion = config.use_fusion
        self.use_uncertainty = config.use_uncertainty

        self.use_diff_decoder = 'DifDe' in self.token_type
        self.base_token_type = self.token_type.replace('-DifDe', '')

        super().__init__(config)

        self.model = InteractVLMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        print(f'config type {type(config)}')
        self.cam_encoder_type = config.cam_encoder_type

        self.human_3d_contact_predictor, self.object_3d_afford_predictor, self.object_3d_contact_predictor = None, None, None
        print(f'Model configs: {config}')
        print(f'Adding extra modules for InteractVLM')
        if self.base_token_type in ['Gen-Hu-Obj', 'Gen-Int']:
            print(f'\t---> Adding AttentionSplitter for token_type: {self.base_token_type}')
            self.attention_splitter = AttentionSplitter()
        if self.hC_loss_weight > 0:
            print(f'\t---> Adding HumanContact3DPredictor for hC_loss_weight: {self.hC_loss_weight}')
            self.human_3d_contact_predictor = HumanContact3DPredictor(self.hC_sam_view_type, self.multiview_channels)
        if self.oC_loss_weight > 0:
            print(f'\t---> Adding ObjectAfford3DPredictor for oC_loss_weight: {self.oC_loss_weight}')
            self.object_3d_afford_predictor = ObjectPCAfford3DPredictor(self.oC_sam_view_type, self.multiview_channels)
            print(f'\t---> Adding ObjectContact3DPredictor for oC_loss_weight: {self.oC_loss_weight}')
            self.object_3d_contact_predictor = ObjectMeshContact3DPredictor(self.oC_sam_view_type, self.multiview_channels)
        if self.multiview_cam_cond:
            print(f'\t---> Adding CamPoseEncoder {self.cam_encoder_type} for multiview_cam_cond: {self.multiview_cam_cond}')
            if self.cam_encoder_type == 'simple':
                self.cam_pose_encoder = CamPoseEncoder()
            elif self.cam_encoder_type == 'view_index':
                self.cam_pose_encoder = ViewIndexCamPoseEncoder(num_views=self.multiview_channels)
            elif self.cam_encoder_type == 'vi_v1':
                self.cam_pose_encoder = VIv1CamPoseEncoder(num_views=self.multiview_channels)

        if train_from_LISA or train_from_LLAVA:
            self.combined_loss_fn = CombinedLoss(
                self.human_3d_contact_predictor,
                self.object_3d_afford_predictor,
                self.object_3d_contact_predictor,
                bce_loss_weight=self.bce_loss_weight,
                bce_loss_alpha=self.bce_loss_alpha,
                dice_loss_weight=self.dice_loss_weight,
                dice_loss_scale=self.dice_loss_scale,
                hC_loss_weight=self.hC_loss_weight,
                oC_loss_weight=self.oC_loss_weight,
                use_uncertainty=self.use_uncertainty,
            )

        self.uncert_initial_weights = None
        self.fusion_initial_weights = None

        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(pixel_values[i])
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.stack(image_embeddings_list, dim=0)

        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def process_embeddings(self, embedding, cam_params, token):

        # Make the llava embeding view aware
        if self.multiview_cam_cond:
            if self.cam_encoder_type == 'simple':
                view_embedding = self.cam_pose_encoder(cam_params)
                embedding += view_embedding # Add camera pose embedding
            elif self.cam_encoder_type == 'view_index' or self.cam_encoder_type == 'vi_v1':
                view_encodings = []
                for view_idx in range(self.multiview_channels):
                    view_enc = self.cam_pose_encoder(cam_params[[view_idx]], view_idx)
                    view_encodings.append(view_enc)
                view_encodings = torch.stack(view_encodings, dim=1)  # [B, V, D]
                # Combine LLaVA embeddings with view-specific information
                embedding = embedding * view_encodings

        if self.base_token_type == 'Gen':
            return embedding
        
        if token == self.hseg_token_idx:
            human_emb, _ = self.attention_splitter(embedding)
            return human_emb
        elif token == self.oseg_token_idx:
            _, object_emb = self.attention_splitter(embedding)
            return object_emb
        else:  # general segmentation
            return embedding

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        gt_contact_3d_list: List[torch.Tensor],
        cam_params: torch.FloatTensor,
        resize_list: List[tuple],
        ds_name_list: List[str],
        mask_paths_list: List[str],
        inference: bool = False,
        **kwargs,
    ):
        # images: [B, MultiView, 3, H, W]
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

        seg_mask = (input_ids == self.seg_token_idx)

        if self.base_token_type in ['Gen-Hu-Obj', 'Gen-Int']:
            hseg_mask = (input_ids == self.hseg_token_idx)
            oseg_mask = (input_ids == self.oseg_token_idx)
            seg_token_mask = seg_mask | hseg_mask | oseg_mask
        else:
            seg_token_mask = seg_mask

        # Remove the first column (corresponding to the image token)
        seg_token_mask = seg_token_mask[:, 1:]

        seg_token_mask = torch.cat([
            seg_token_mask,
            torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
        ], dim=1)
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat([
            torch.zeros((seg_token_mask.shape[0], self.img_emb_len)).bool().cuda(),
            seg_token_mask
        ], dim=1)

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        tokens = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            seg_indices = seg_token_mask[i].nonzero().squeeze(-1)
            # Adjust indices to account for the image token and the fact that we started from input_ids[:, 1:]
            adjusted_indices = seg_indices - self.img_emb_len + 1
            # Filter out non-positive indices (which would correspond to the image token or the padding)
            valid_indices = adjusted_indices[adjusted_indices > 0]
            if valid_indices.numel() > 0:
                tokens.append(input_ids[i, valid_indices[0]].cpu().item())
            else:
                tokens.append(None)
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks, gt_masks, uncertainty_maps = [], [], []
        llava_features = output_hidden_states[-1] if self.use_fusion else None

        for i in range(len(pred_embeddings)):
            pred_emb = pred_embeddings[i].unsqueeze(1)
            if self.multiview_channels > 1:
                pred_emb = pred_emb.repeat(1, self.multiview_channels, 1)
            processed_embedding = self.process_embeddings(pred_emb, cam_params[i], tokens[i])
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=processed_embedding,
            )
            sparse_embeddings = sparse_embeddings.to(processed_embedding.dtype)
            
            low_res_masks, iou_predictions = self.model.visual_model(
                image_embeddings=image_embeddings[i],
                llava_features=llava_features[i].unsqueeze(0) if self.use_fusion else None,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                ds_name=ds_name_list[i],
            )
            
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
            gt_masks.append(masks_list[i][:, 0])

            if self.use_uncertainty:
                uncertainty_map = self.model.visual_model.uncertainty(image_embeddings[i])
                uncertainty_map = F.interpolate(uncertainty_map, size=label_list[i].shape[-2:], mode='bilinear', align_corners=False)
                uncertainty_maps.append(uncertainty_map.squeeze(0))

        model_output = output
        
        # Apply sigmoid to the predicted masks if the view type is HM for objects
        for idx, ds_name in enumerate(ds_name_list):
            if 'oafford' in ds_name and 'HM' in self.oC_sam_view_type:
                valid_mask = gt_masks[idx] != IGNORE_LABEL
                pred_masks[idx][valid_mask] = pred_masks[idx][valid_mask].sigmoid()

        if inference:
            result = {
                "gt_masks": gt_masks,
                "pred_masks": pred_masks,
            }
            if self.hC_loss_weight > 0:
                pred_human_3d_contacts = self.human_3d_contact_predictor(pred_masks, ds_name_list)
                result["pred_human_3d_contact"] = pred_human_3d_contacts
            if self.oC_loss_weight > 0:
                pred_object_3d_contacts = self.object_3d_contact_predictor(pred_masks, ds_name_list, mask_paths_list)
                result["pred_object_3d_contact"] = pred_object_3d_contacts
                pred_object_3d_afford = self.object_3d_afford_predictor(pred_masks, ds_name_list, mask_paths_list)
                result["pred_object_3d_afford"] = pred_object_3d_afford
            if self.use_uncertainty:
                result["uncertainty_maps"] = uncertainty_maps

            return result

        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        combined_loss, mask_bce_loss, mask_dice_loss, mask_l2_loss, hC_loss, oA_loss, oC_loss, uncertainty_loss = \
            self.combined_loss_fn(
                pred_masks,
                gt_masks,
                gt_contact_3d_list,
                mask_paths_list,
                ds_name_list,
                uncertainty_maps if self.use_uncertainty else None,
            )
        mask_loss = mask_bce_loss + mask_dice_loss + mask_l2_loss + uncertainty_loss 
        loss = ce_loss + combined_loss

        results = {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_l2_loss": mask_l2_loss,
            "mask_loss": mask_loss,
            "hC_loss": hC_loss,
            "oA_loss": oA_loss,
            "oC_loss": oC_loss,
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
        }
        if self.use_uncertainty:
            results["uncertainty_maps"] = uncertainty_maps

        return results

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        cam_params,
        resize_list,
        original_size_list,
        lift2d_dict_path=None,
        contact_type='hcontact',
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = torch.zeros_like(output_ids, dtype=torch.bool, device=output_ids.device)
            seg_mask = (output_ids == self.seg_token_idx)
            
            if self.base_token_type in ['Gen-Hu-Obj', 'Gen-Int']:
                hseg_mask = (output_ids == self.hseg_token_idx)
                oseg_mask = (output_ids == self.oseg_token_idx)
                seg_token_mask = seg_mask | hseg_mask | oseg_mask
            else:
                seg_token_mask = seg_mask

            seg_token_mask = seg_token_mask[:, 1:]
            seg_token_mask = torch.cat([
                torch.zeros((seg_token_mask.shape[0], self.img_emb_len)).bool().cuda(),
                seg_token_mask
            ], dim=1)

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            tokens = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
                seg_indices = seg_token_mask[i].nonzero().squeeze(-1)
                adjusted_indices = seg_indices - self.img_emb_len + 1
                valid_indices = adjusted_indices[adjusted_indices > 0]
                if valid_indices.numel() > 0:
                    tokens.append(output_ids[i, valid_indices[0]].cpu().item())
                else:
                    tokens.append(None)
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            uncertainty_maps = []
            llava_features = output_hidden_states[-1] if self.use_fusion else None

            for i in range(len(pred_embeddings)):
                pred_emb = pred_embeddings[i].unsqueeze(1)
                if self.multiview_channels > 1:
                    pred_emb = pred_emb.repeat(1, self.multiview_channels, 1)
                processed_embedding = self.process_embeddings(pred_emb, cam_params[i], tokens[i])
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=processed_embedding,
                )

                sparse_embeddings = sparse_embeddings.to(processed_embedding.dtype)
                
                low_res_masks, _ = self.model.visual_model(
                    image_embeddings=image_embeddings[i],
                    llava_features=llava_features[i].unsqueeze(0) if self.use_fusion else None,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    ds_name=contact_type,
                )
                
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

                if self.use_uncertainty:
                    uncertainty_map = self.model.visual_model.uncertainty(image_embeddings[i].unsqueeze(0))
                    uncertainty_map = F.interpolate(uncertainty_map, size=original_size_list[i], mode='bilinear', align_corners=False)
                    uncertainty_maps.append(uncertainty_map.squeeze(0))

        
        if 'oafford' in contact_type and 'HM' in self.oC_sam_view_type:
            for idx, pred_mask in enumerate(pred_masks):
                valid_mask = pred_mask != IGNORE_LABEL
                pred_mask[valid_mask] = pred_mask[valid_mask].sigmoid()
        
        pred_contact_3d = None

        if pred_masks[0].shape[0] > 0:
            if self.hC_loss_weight > 0 and 'hcontact' in contact_type:
                pred_contact_3d = self.human_3d_contact_predictor(pred_masks)
                # TODO: check if the funnction call is correct
            elif self.oC_loss_weight > 0 and 'ocontact' in contact_type and 'Mesh' in self.oC_sam_view_type:
                pred_contact_3d = self.object_3d_contact_predictor(pred_masks, lift2d_dict_path=lift2d_dict_path)

        result = {
            "output_ids": output_ids,
            "pred_masks": pred_masks,
            "pred_contact_3d": pred_contact_3d,
        }
        if self.use_uncertainty:
            result["uncertainty_maps"] = uncertainty_maps

        return result