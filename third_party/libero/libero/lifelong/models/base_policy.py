import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.data_augmentation import (
    IdentityAug,
    TranslationAug,
    ImgColorJitterAug,
    ImgColorJitterGroupAug,
    BatchWiseImgColorJitterAug,
    DataAugGroup,
)

REGISTERED_POLICIES = {}


def register_policy(policy_class):
    """将一个策略类注册到注册表中。"""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_POLICIES:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_POLICIES[policy_name] = policy_class


def get_policy_class(policy_name):
    """从注册表中获取策略类。"""
    if policy_name.lower() not in REGISTERED_POLICIES:
        raise ValueError(
            "Policy class with name {} not found in registry".format(policy_name)
        )
    return REGISTERED_POLICIES[policy_name.lower()]


def get_policy_list():
    return REGISTERED_POLICIES


class PolicyMeta(type):
    """用于注册环境的元类"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # 列出所有不应在此处注册的策略。
        _unregistered_policies = ["BasePolicy"]

        if cls.__name__ not in _unregistered_policies:
            register_policy(cls)
        return cls


class BasePolicy(nn.Module, metaclass=PolicyMeta):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.shape_meta = shape_meta

        policy_cfg = cfg.policy

        # 为 rgb 输入添加数据增强
        color_aug = eval(policy_cfg.color_aug.network)(
            **policy_cfg.color_aug.network_kwargs
        )

        policy_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
            "all_shapes"
        ][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(policy_cfg.translation_aug.network)(
            **policy_cfg.translation_aug.network_kwargs
        )
        self.img_aug = DataAugGroup((color_aug, translation_aug))

    def forward(self, data):
        """
        用于训练的前向函数。
        """
        raise NotImplementedError

    def get_action(self, data):
        """
        获取策略动作的 api。
        """
        raise NotImplementedError

    def _get_img_tuple(self, data):
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode:  # 应用数据增强
            if self.cfg.train.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # 添加时间维度
            )
            data["task_emb"] = data["task_emb"].squeeze(1)
        return data

    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        return loss

    def reset(self):
        """
        清除策略的所有“历史记录”（如果存在）。
        """
        pass
