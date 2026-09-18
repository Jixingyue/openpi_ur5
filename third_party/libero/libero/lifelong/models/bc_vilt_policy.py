import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from einops import rearrange, repeat
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens


###############################################################################
#
# 一个 ViLT 策略
#
###############################################################################


def reshape_transform(tensor, h, w):
    B, _, E = tensor.shape
    result = tensor[:, 1 : 1 + h * w, :].reshape(B, h, w, E)
    return result.permute(0, 3, 1, 2)


class BCViLTPolicy(BasePolicy):
    """
    输入：(o_{t-H}, ... , o_t)
    输出：a_t 或 a_t 的分布
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. 编码图像
        embed_size = policy_cfg.embed_size

        transformer_input_sizes = []
        self.image_encoders = {}

        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.embed_size = embed_size
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )
        num_patches = sum([x.num_patches for x in self.encoders])

        ### 2. 编码语言（空间）
        policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder_spatial = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )

        ### 3. 定义位置嵌入、模态嵌入以及用于汇总的空间 token
        spatial_token = nn.Parameter(torch.randn(1, 1, embed_size))  # SPATIAL_TOKEN
        patch_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_size))
        modality_embed = nn.Parameter(
            torch.randn(1, len(self.encoders) + 1, embed_size)
        )  # PATCH_TOKENS + SENTENCE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("patch_pos_embed", patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # 用于选择模态嵌入
        modality_idx = []
        for i, x in enumerate(self.encoders):
            modality_idx += [i] * x.num_patches
        modality_idx += [modality_idx[-1] + 1]  # 用于句子嵌入
        self.modality_idx = torch.LongTensor(modality_idx).to(cfg.device)

        ### 4. 定义空间 transformer
        self.spatial_transformer = TransformerDecoder(
            input_size=embed_size,
            num_layers=policy_cfg.spatial_transformer_num_layers,
            num_heads=policy_cfg.spatial_transformer_num_heads,
            head_output_size=policy_cfg.spatial_transformer_head_output_size,
            mlp_hidden_size=policy_cfg.spatial_transformer_mlp_hidden_size,
            dropout=policy_cfg.spatial_transformer_dropout,
        )

        if policy_cfg.spatial_down_sample:
            temporal_embed_size = policy_cfg.spatial_down_sample_embed_size
            self.spatial_down_sample = nn.Linear(embed_size, temporal_embed_size)
        else:
            temporal_embed_size = embed_size

        ### 5. 编码额外信息（例如夹爪、joint_state）
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=temporal_embed_size,
        )
        num_extra = self.extra_encoder.num_extra

        ### 6. 编码语言（时序），这也将充当 TEMPORAL_TOKEN
        policy_cfg.language_encoder.network_kwargs.output_size = temporal_embed_size
        self.language_encoder_temporal = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )

        ### 7. 定义时序 transformer
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = (
            temporal_embed_size
        )
        self.temporal_position_encoding_fn = eval(
            policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        self.temporal_transformer = TransformerDecoder(
            input_size=temporal_embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
        )

        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = temporal_embed_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )

        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

        ### 8. 用于注意力可视化的 reshape 变换
        self.reshape_transform = lambda x: reshape_transform(
            x, self.encoders[0].h, self.encoders[1].w
        )

    def spatial_encode(self, data):
        # 1. 编码图像
        img_encoded = []
        for img_name in self.image_encoders.keys():
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        data["obs"][img_name], self.image_encoders[img_name]["encoder"]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # 添加 img_h: (B, T, num_patches, E)
        img_encoded = torch.cat(img_encoded, -2)  # (B, T, 2*num_patches, E)
        img_encoded += self.patch_pos_embed.unsqueeze(0)  # (B, T, 2*num_patches, E)
        B, T = img_encoded.shape[:2]

        # 2. 编码 task_emb
        text_encoded = self.language_encoder_spatial(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)

        # 3. 拼接 img + text 嵌入，然后添加模态嵌入
        img_text_encoded = torch.cat(
            [img_encoded, text_encoded], -2
        )  # (B, T, 2*num_patches+1, E)
        img_text_encoded += self.modality_embed[
            None, :, self.modality_idx, :
        ]  # 与上面相同

        # 4. 添加空间 token
        spatial_token = self.spatial_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (B, T, 1, E)
        encoded = torch.cat([spatial_token, img_text_encoded], -2)  # (B, T, :, E)

        # 5. 传入 transformer
        encoded = rearrange(encoded, "b t n e -> (b t) n e")  # (B*T, :, E)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # 提取空间 token 作为 o_t 处的汇总
        out = self.spatial_down_sample(out).view(B, T, 1, -1)  # (B, T, 1, E')

        # 6. 编码额外信息
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E')

        # 7. 编码语言，将其视为动作 token
        text_encoded_ = self.language_encoder_temporal(data)  # (B, E')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E')
        out = torch.cat([text_encoded_, out, extra], -2)  # (B, T, :, E')
        return out

    def temporal_encode(self, x):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def forward(self, data):
        x = self.spatial_encode(data)  # (B, T, E)
        x = self.temporal_encode(x)  # (B, T, E)
        dist = self.policy_head(x)
        return dist

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)
            self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
            x = self.temporal_encode(x)
            dist = self.policy_head(x[:, -1])
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []
