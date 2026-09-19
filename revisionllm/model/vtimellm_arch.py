import torch
import torch.nn as nn
from revisionllm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, MEMORY_TOKEN_INDEX
from abc import ABC, abstractmethod
# from revisionllm.model.adapter.cross_attn import MLP, CrossAttn
from revisionllm.model.adapter.transformer import ClipEncoder, PositionEmbeddingSine
from revisionllm.model.adapter.audio_fusion import AudioFusion, load_audio_fusion_weights
from einops import rearrange


class VTimeLLMMetaModel:

    def initialize_audio_modules(self, model_args):
        """
        ClipCraft: optional CLAP audio branch fused additively into the raw CLIP
        frame features *before* any projector (linear mm_projector, ClipEncoder
        adapter, hierarchy CLS pooling). Guarded with ``hasattr`` so that a
        second call (train.py calls initialize_* twice) never resets weights
        that ``load_lora`` already loaded from ``non_lora_trainables.bin``.
        """
        if not getattr(model_args, 'audio_fusion', False):
            return
        if hasattr(self, 'audio_fusion'):
            return
        self.audio_fusion = AudioFusion(
            audio_dim=getattr(model_args, 'audio_dim', 512),
            visual_dim=getattr(model_args, 'adapter_input_dim', 768),
            gate_init=getattr(model_args, 'audio_gate_init', 0.0),
            dropout=getattr(model_args, 'audio_dropout', 0.0),
        )
        pretrain = getattr(model_args, 'pretrain_audio_fusion', None)
        if pretrain is not None:
            load_audio_fusion_weights(self.audio_fusion, pretrain)

    def initialize_vision_modules(self, model_args):
        assert not model_args.clip_adapter or not model_args.cross_attn, "both clip_adapter and cross_attn cannot be true"
        self.initialize_audio_modules(model_args)
        self.pretrain_clip_adapter = model_args.pretrain_clip_adapter
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.clip_adapter = model_args.clip_adapter
        self.clip_adapter_text = model_args.clip_adapter_text
        self.clip_adapter_feature = model_args.clip_adapter_feature
        self.hierarchy = model_args.hierarchy
        if not hasattr(self, 'mm_projector'):
            if self.clip_adapter:
                self.mm_projector = ClipEncoder(hidden_size=self.config.hidden_size,
                                                clip_adapter_text=model_args.clip_adapter_text,
                                                cross_attn=model_args.cross_attn,
                                                hierarchy=self.hierarchy,
                                                clip_adapter_feature=self.clip_adapter_feature)
                if model_args.pretrain_clip_adapter is not None:
                    mm_projector_weights = torch.load(model_args.pretrain_clip_adapter, map_location='cpu')

                    def get_wc(weights, keyword):
                        return_dict = {}
                        for k, v in weights.items():
                            if keyword in k and f'{keyword}.{keyword}' not in k:
                                return_dict[k.split(keyword + '.')[1]] = v
                            else:
                                return_dict[keyword + '.' + k.split(keyword + '.')[2]] = v
                        return return_dict

                    self.mm_projector.load_state_dict(get_wc(mm_projector_weights, 'mm_projector'), strict=False)
                    print("load clip adapter:", model_args.pretrain_clip_adapter)
            else:
                self.mm_projector = nn.Linear(model_args.adapter_input_dim, self.config.hidden_size)

                if pretrain_mm_mlp_adapter is not None:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                    def get_w(weights, keyword):
                        return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                    self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
                    print("load mlp:", pretrain_mm_mlp_adapter)

        # ClipCraft: guard so a second initialize_vision_modules() call does not
        # re-create cross_attn and discard weights loaded by load_lora().
        if model_args.cross_attn and not hasattr(self, 'cross_attn'):
            self.cross_attn = ClipEncoder(hidden_size=self.config.hidden_size,
                                          clip_adapter_text=model_args.clip_adapter_text,
                                          cross_attn=model_args.cross_attn and model_args.pretrain_clip_adapter is None,
                                          hierarchy=self.hierarchy,
                                          clip_adapter_feature=self.clip_adapter_feature)
            if model_args.pretrain_clip_adapter is not None:
                mm_projector_weights = torch.load(model_args.pretrain_clip_adapter, map_location='cpu')

                def get_wc(weights, keyword):
                    return_dict = {}
                    for k, v in weights.items():
                        if keyword in k and f'{keyword}.{keyword}' not in k:
                            return_dict[k.split(keyword + '.')[1]] = v
                        else:
                            return_dict[keyword+'.'+k.split(keyword + '.')[2]] = v
                    return return_dict

                self.cross_attn.load_state_dict(get_wc(mm_projector_weights, 'mm_projector'), strict=False)
                print("load clip adapter:", model_args.pretrain_clip_adapter)
        if model_args.clip_adapter_feature == 'alternate' and not hasattr(self, 'alternate_layer_norm'):
            self.alternate_layer_norm = nn.LayerNorm(self.config.hidden_size)

class VTimeLLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def fuse_audio(self, images, audio_feats):
        """
        ClipCraft: additive audio fusion on raw CLIP features. Identity when the
        model has no audio branch or no audio was provided. Handles the
        list-of-tensors batch form used by the collator for ragged batches.
        """
        model = self.get_model()
        if audio_feats is None or not hasattr(model, 'audio_fusion'):
            return images
        fusion = model.audio_fusion
        if type(images) is list:
            if type(audio_feats) is not list or len(audio_feats) != len(images):
                raise ValueError("fuse_audio: images is a list but audio_feats is not a matching list")
            return [fusion(im, au) for im, au in zip(images, audio_feats)]
        return fusion(images, audio_feats)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, query_feats, visual_memory, prefix_memory,iteration_step,
        audio_feats=None,
    ):
        # print(position_ids, attention_mask)
        # if past_key_values:
        #     print(past_key_values[-1][-1].shape)
        # print(input_ids.shape, position_ids.shape, attention_mask.shape, past_key_values.shape, images)
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                if self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # ClipCraft: fuse audio into the raw frame features. Every downstream
        # path (linear projector, ClipEncoder, hierarchy CLS) then sees the
        # audio-augmented features with unchanged tensor shapes.
        images = self.fuse_audio(images, audio_feats)

        if type(images) is list:
            # for image in images:
            #     print('image.shape', image.shape)
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().mm_projector(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            if self.get_model().clip_adapter and not hasattr(self.get_model(), 'cross_attn'):
                if self.get_model().hierarchy and self.get_model().clip_adapter_feature == 'alternate' and iteration_step % 2 == 1:  # Temporal
                    image_features = self.get_model().mm_projector(images, query_feats[0], query_feats[1],iteration_step)
                elif self.get_model().hierarchy: #CLS
                    images_h = rearrange(images, 'b v t d -> (b v) t d')
                    query_feats_h = query_feats[0][:, None].repeat(1, images.shape[1], 1, 1)
                    query_feats_h = rearrange(query_feats_h, 'b v t d -> (b v) t d')
                    query_masks_h = query_feats[1][:, None].repeat(1, images.shape[1], 1)
                    query_masks_h = rearrange(query_masks_h, 'b v d -> (b v) d')
                    image_features = self.get_model().mm_projector(images_h, query_feats_h, query_masks_h,iteration_step)
                    image_features = rearrange(image_features, '(b v) 1 d -> b v d', b=images.shape[0])
                else:
                    image_features = self.get_model().mm_projector(images, query_feats[0], query_feats[1],iteration_step)
            else:
                image_features = self.get_model().mm_projector(images)

        if hasattr(self.get_model(), 'cross_attn') and not self.get_model().clip_adapter:
            if self.get_model().clip_adapter_feature=='alternate' and iteration_step%2==1:#Temporal
                # image_features = self.get_model().cross_attn(image_features, query_feats[0], query_feats[1], iteration_step)
                pass
            elif self.get_model().hierarchy:
                if self.get_model().pretrain_clip_adapter is not None:
                    image_features = images
                images_h = rearrange(image_features, 'b v t d -> (b v) t d')
                query_feats_h = query_feats[0][:,None].repeat(1, image_features.shape[1], 1, 1)
                query_feats_h = rearrange(query_feats_h, 'b v t d -> (b v) t d')
                query_masks_h = query_feats[1][:,None].repeat(1, image_features.shape[1], 1)
                query_masks_h = rearrange(query_masks_h, 'b v d -> (b v) d')
                image_features = self.get_model().cross_attn(images_h, query_feats_h, query_masks_h, iteration_step)
                image_features = rearrange(image_features, '(b v) 1 d -> b v d', b=images.shape[0])
            else:
                if self.get_model().pretrain_clip_adapter is not None:
                    image_features = images
                image_features = self.get_model().cross_attn(image_features, query_feats[0], query_feats[1], iteration_step)

        if self.get_model().clip_adapter_feature=='alternate':
            image_features = self.get_model().alternate_layer_norm(image_features)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            if visual_memory is None:
                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            else:
                image_token_indices = ([-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]])

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # split into chunks
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            if visual_memory is None:
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        if cur_image_features.dim() == 1:
                            cur_image_features = cur_image_features[None]
                        cur_new_input_embeds.append(cur_image_features) #inject image
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            else:
                #First Chunk
                cur_new_input_embeds.append(cur_input_embeds_no_im[0])
                cur_new_labels.append(cur_labels_noim[0])
                #Video
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds.append(cur_image_features)  # inject image
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                #Second Chunk
                cur_new_input_embeds.append(cur_input_embeds_no_im[1])
                cur_new_labels.append(cur_labels_noim[1])
                #Memory
                vis_mem = visual_memory[:,None] if visual_memory.dim()==2 else visual_memory
                memory_features = torch.cat([self.get_model().get_input_embeddings()(prefix_memory),
                                             self.get_model().mm_projector(vis_mem)],1)
                cur_memory_features = memory_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_memory_features)  # inject memory
                cur_new_labels.append(torch.full((cur_memory_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                #Third Chunk
                if len(cur_input_embeds_no_im)==3:
                    cur_new_input_embeds.append(cur_input_embeds_no_im[2])
                    cur_new_labels.append(cur_labels_noim[2])
                else:
                    print("cur_new_input_embeds len is 2: ", cur_input_ids)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000, 
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None
        # print(position_ids, attention_mask)
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
