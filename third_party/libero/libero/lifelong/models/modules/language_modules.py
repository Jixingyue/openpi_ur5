"""
该文件包含对语言任务嵌入进行编码的模块。
"""
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """
    直接输出预训练任务嵌入的虚拟编码器
    """

    def __init__(self, dummy=True):
        super().__init__()

    def forward(self, data):
        """
        数据:
            task_emb: (B, E)
        """
        h = data["task_emb"]  # (B, L, H)
        return h


class MLPEncoder(nn.Module):
    """
    对任务嵌入进行编码

    h = f(e)，其中
        e：来自大模型的预训练任务嵌入
        h：潜在嵌入 (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        数据:
            task_emb: (B, E)
        """
        h = self.projection(data["task_emb"])  # (B, H)
        return h
