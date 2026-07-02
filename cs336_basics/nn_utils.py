import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        ''' 
        构建 RMSNorm 模块。该函数应接受以下参数：
        d_model: int  模型的隐藏维度
        eps: float = 1e-5  用于数值稳定性的 Epsilon 值
        device: torch.device | None = None  存储参数的设备
        dtype: torch.dtype | None = None  参数的数据类型
        '''
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        处理形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回相同形状的张量。
        '''
        # 实现 RMSNorm 模块的前向传播
        return x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt() * self.weights

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    '''
    对输入张量的指定维度应用 softmax 函数。
    '''
    return (x - x.max(dim=dim, keepdim=True).values).exp() / (x - x.max(dim=dim, keepdim=True).values).exp().sum(dim=dim, keepdim=True)

def silu(x: torch.Tensor) -> torch.Tensor:
    '''
    对输入张量应用 SiLU（Sigmoid Linear Unit）激活函数。
    '''
    return x * torch.sigmoid(x)

def cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    对输入张量 x 应用交叉熵损失函数，计算与目标张量 y 的损失。
    '''
    log_softmax = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return -torch.mean(log_softmax[torch.arange(len(y)), y], dim=-1)
    # return -(log_softmax[torch.arange(len(y)), y].mean())
    # return -(log_softmax[torch.arange(len(y)), y]).mean()

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    对输入张量 Q、K、 V 应用注意力机制，计算注意力权重并返回注意力输出。
    '''
    scores = Q @ K.mT / math.sqrt(K.shape[-1])
    # True 表示允许关注（保留分数），False 表示不允许关注（将分数设为负无穷，遮掉该位置）
    scores = scores.masked_fill(~mask, -torch.inf)
    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ V

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to query or key tensor.
    """
    d_k_half = d_k // 2
    device = in_query_or_key.device

    dim_indices = torch.arange(d_k_half, device=device)
    theta_i = theta ** (-2 * dim_indices / d_k)

    # Get positions
    if token_positions.dim() > 1:
        positions = token_positions.reshape(-1)[:in_query_or_key.shape[-2]]
    else:
        positions = token_positions

    angles = torch.outer(positions.float(), theta_i)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # Expand to match input dimensions: (..., seq, d_k_half)
    ndim = in_query_or_key.dim()
    cos_vals = cos_vals.view(*(1,) * (ndim - 2), in_query_or_key.shape[-2], d_k_half)
    sin_vals = sin_vals.view(*(1,) * (ndim - 2), in_query_or_key.shape[-2], d_k_half)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd = x_even * sin_vals + x_odd * cos_vals

    out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    out = out.view(*in_query_or_key.shape)

    return out