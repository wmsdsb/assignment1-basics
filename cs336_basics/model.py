import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.nn_utils import silu, attention, run_rope

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int 输入的最终维度
        out_features: int 输出的最终维度
        device: torch.device | None = None 存储参数的设备
        dtype: torch.dtype | None = None 参数的数据类型
        '''
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights)
        # self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # self.weights = nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        输入: torch.Tensor 形状为 (batch_size, in_features) 的张量。
        返回: torch.Tensor 形状为 (batch_size, out_features) 的张量。
        '''
        return x @ self.weights.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int 词汇表大小
        embedding_dim: int 嵌入维度
        device: torch.device | None = None 存储参数的设备
        dtype: torch.dtype | None = None 参数的数据类型
        '''
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights)
        # self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        查找给定 token ID 对应的embedding 向量。
        token_ids: torch.Tensor 形状为 (batch_size, sequence_length) 的整数张量，表示 token ID。
        返回: torch.Tensor 形状为 (batch_size, sequence_length, embedding_dim) 的嵌入向量。
        '''
        # 从 embedding 权重矩阵中提取对应 token ID 的向量
        return self.weights[token_ids]


class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int 输入的最终维度
        out_features: int 输出的最终维度
        device: torch.device | None = None 存储参数的设备
        dtype: torch.dtype | None = None 参数的数据类型
        '''
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1)
        self.w2 = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w2)
        self.w3 = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        输入: torch.Tensor 形状为 (batch_size, in_features) 的张量。
        返回: torch.Tensor 形状为 (batch_size, out_features) 的张量。
        '''
        out = silu(x @ self.w1.T)
        out = out * (x @ self.w3.T)
        out = out @ self.w2.T
        return out

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: float 角度参数
        d_k: int 键值维度
        max_seq_len: int 最大序列长度
        device: torch.device | None = None 存储参数的设备
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        输入: torch.Tensor 形状为 (batch_size, sequence_length, d_model) 的张量。
        token_positions: torch.Tensor 形状为 (batch_size, sequence_length) 的整数张量，表示 token位置。
        返回: torch.Tensor 形状为 (batch_size, sequence_length, d_model) 的张量。
        '''
        ''' θ_i = theta^(-2i/d)    对于 i = 0, 1, ..., d/2-1(维度)
    
        对每个位置 pos: 
        x_偶数位 ← x_偶数位 * cos(pos * θ_i) - x_奇数位 * sin(pos * θ_i)
        x_奇数位 ← x_偶数位 * sin(pos * θ_i) + x_奇数位 * cos(pos * θ_i)
        '''
        seq_len = x.shape[1]
        d_k = x.shape[2]
        d_k_half = d_k // 2

        dim_indices = torch.arange(d_k_half, device=self.device)
        theta_i = self.theta ** (-2 * dim_indices / self.d_k)

        pos = torch.arange(seq_len, device=self.device)

        # 外积 (seq_len, d_k_half)
        angles = torch.outer(pos, theta_i)

        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        # 将 cos_vals 和 sin_vals 扩展到 (batch_size, seq_len, d_k_half)
        cos_vals = cos_vals.unsqueeze(0).repeat(x.shape[0], 1, 1)
        sin_vals = sin_vals.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # 计算新的 x 值
        x_even = x[:, :, 0::2] * cos_vals - x[:, :, 1::2] * sin_vals    # 偶数位
        x_odd = x[:, :, 0::2] * sin_vals + x[:, :, 1::2] * cos_vals     # 奇数位
        
        # 合并偶数位和奇数位
        x = torch.stack([x_even, x_odd], dim=-1)
        x = x.view(x.shape[0], x.shape[1], d_k)
               
        return x
        # x_rotated = torch.zeros_like(x)
        # x_rotated[:, :, 0::2] = x_even
        # x_rotated[:, :, 1::2] = x_odd
        
        # return x_rotated

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        # 定义 Q, K, V, O 四个投影矩阵
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_model = d_model
        self.num_heads = num_heads
        self.Q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.Q)
        self.K = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.K)
        self.V = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.V)
        self.O = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.O)

    def forward(self, x, token_positions: torch.Tensor = None):
        # 1. 线性投影
        # 2. 拆头
        # 3. attention
        # 4. 合并头
        # 5. 输出投影
        batch_size, seq_len, _ = x.shape
        d_model = self.d_model
        num_heads = self.num_heads
        d_k = d_model // num_heads

        Q = x @ self.Q.T
        K = x @ self.K.T
        V = x @ self.V.T
        
        Q = Q.view(batch_size, seq_len, num_heads, d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, num_heads, d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, num_heads, d_k).permute(0, 2, 1, 3)

        if token_positions is not None and hasattr(self, 'theta'):
            Q = run_rope(d_k, self.theta, self.max_seq_len, Q, token_positions)
            K = run_rope(d_k, self.theta, self.max_seq_len, K, token_positions)
        

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        attention_out = attention(Q, K, V, causal_mask)
        attention_out = attention_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attention_out = attention_out @ self.O.T
        return attention_out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.attention = MultiheadAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.dtype = dtype
