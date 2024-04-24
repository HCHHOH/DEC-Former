import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

from utils import norm_Adj

class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)
        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))

class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, sym_norm_Geo_matrix, sym_norm_Sem_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.sym_norm_Geo_matrix = sym_norm_Geo_matrix  # (N, N)
        self.sym_norm_Sem_matrix = sym_norm_Sem_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta1 = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta2 = nn.Linear(in_channels, out_channels, bias=False)
        self.fusion_activation = nn.Sigmoid()
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)
        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        # 区别：mul函数，将sym_norm_Adj_matrix 和 spatial_attention 进行逐元素相乘

        # attention 和 x 逐元素相乘
        spatial_score_adj = torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)
        spatial_score_adj = self.Theta(spatial_score_adj).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)

        # spatial_score_geo = torch.matmul(self.sym_norm_Geo_matrix.mul(spatial_attention), x)
        # spatial_score_geo = self.Theta1(spatial_score_geo).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)
        #
        # spatial_score_sem = torch.matmul(self.sym_norm_Sem_matrix.mul(spatial_attention), x)
        # spatial_score_sem = self.Theta2(spatial_score_sem).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)
        #
        # # 计算门控值
        # # g = self.activation(spatial_score_geo + spatial_score_sem)
        # g = 0.2
        # spatial_score_adj = g * spatial_score_geo + (1 - g) * spatial_score_sem
        spatial_score_adj = F.relu(spatial_score_adj)

        # F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape(
        #     (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        return spatial_score_adj
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)

class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        return self.dropout(F.relu(self.gcn(x)))

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(2, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

def search_index(max_len, num_of_depend, num_for_predict,points_per_hour, units):
    '''
    Parameters
    ----------
    max_len: int, length of all encoder input
    num_of_depend: int,
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx

def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)

def FourierAttention(q, k, v, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    # B, N, H, T1, E = q.shape
    # B, N, H, T2, E = k.shape
    d_k = q.size(-1)
    T1 = q.size(-2)
    T2 = k.size(-2)

    xq = q.permute(0, 1, 2, 4, 3)
    xk = k.permute(0, 1, 2, 4, 3)
    xv = v.permute(0, 1, 2, 4, 3)

    xq_ft_ = torch.fft.fft(xq, dim=-1, norm='ortho')
    xk_ft_ = torch.fft.fft(xk, dim=-1, norm='ortho')
    xv_ft_ = torch.fft.fft(xv, dim=-1, norm='ortho')

    xqk_ft = torch.einsum("bnhex,bnhey->bnhxy", xq_ft_, torch.conj(xk_ft_)) / math.sqrt(d_k) # 注意参数

    # 计算注意力分数的绝对值
    xqk_ft_abs = xqk_ft.abs()
    # 如果有 mask，使用 masked_fill_ 进行填充
    if mask is not None:
        # xqk_ft_shape = xqk_ft_abs.shape
        # mask = mask[:, :, :, :xqk_ft_shape[-2], :xqk_ft_shape[-1]]
        xqk_ft_abs = xqk_ft_abs.masked_fill_(mask == 0, -1e9)

    xqk_ft = torch.softmax(xqk_ft_abs, dim=-1)
    if dropout is not None:
        xqk_ft = dropout(xqk_ft)

    att = torch.fft.ifft(xqk_ft, n=T2, dim=-1, norm='ortho')
    att = torch.fft.ifft(att, n=T1, dim=-2, norm='ortho')

    # 将实部 xqk_ft 转换为复数形式，其中虚部设置为零.将实部转换为复数。
    # 这样，在进行矩阵乘法时，PyTorch 能够正确处理复数的运算。
    xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
    xqkv_ft = torch.einsum("bnhxy,bnhey->bnhex", xqk_ft, xv_ft_) # 注意参数

    out = torch.fft.ifft(xqkv_ft, n=T1, dim=-1, norm='ortho').real.permute(0, 1, 2, 4, 3)

    return out, att  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

def ESMAttention(query, key, value, sigma, mask=None, dropout=None):
    '''
    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param sigma:  (batch, N, h, T1)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    B, N, H, T1, E = query.shape
    _, _, _, T2, _ = key.shape
    d_k = query.size(-1)
    scale = 1. / math.sqrt(d_k)

    scores = torch.matmul(query, key.transpose(-2, -1))  # scores: (batch, N, h, T1, T2)

    sigma = torch.sigmoid(sigma * 5) + 1e-5
    sigma = torch.pow(3, sigma) - 1
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, 1, T1)  # B N H L L
    distance = torch.linspace(0, T1 - 1, T1).to(sigma.device).squeeze() # 生成一个从 0 到 T-1 的等差数列，该函数会生成 T 个数
    distance = distance.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0). \
        repeat(sigma.shape[0], sigma.shape[1], sigma.shape[2], sigma.shape[3], 1)  # 1 1 1 1 L  --> B N H L L
    distance = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-distance ** 2 / 2 / (sigma ** 2)) # 通过正态分布概率密度函数计算距离权重
    scores = scores + distance

    attn_mask = TriangularCausalMask(B, T1, device=query.device)
    scores.masked_fill_(attn_mask.mask, -np.inf)

    p_attn = F.softmax(scale * scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)

class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0, domain='Time'):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.sigma_l = nn.Linear(d_model, nb_head)
        self.dropout = nn.Dropout(p=dropout)
        self.domain = domain

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)
        sigma = self.sigma_l(query).transpose(2, 3)
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        # 线性变换-->形状变换-->维度转置
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]
        # apply attention on all the projected vectors in batch
        if self.domain == 'Time':
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        elif self.domain == 'Fourier':
            x, self.attn = FourierAttention(query, key, value, mask=mask, dropout=self.dropout)
        elif self.domain == 'ESM':
            x, self.attn = ESMAttention(query, key, value, sigma, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)
        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        x = self.linears[-1](x)

        return x

class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, trend, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        self.trends = trend
        # self.revin = revin
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x, src_trend):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        B, N, T, D = x.shape
        if self.residual_connection or self.use_LayerNorm:
            seasonal_out = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            if src_trend is not None:
                # seasonal_ratio = x.abs().mean(dim=2) / seasonal_out.abs().mean(dim=2) # （B, N, D）
                # seasonal_ratio = seasonal_ratio.unsqueeze(2).expand(-1, -1, T, -1)

                # trend_enc = self.revin(src_trend, 'norm')
                trend_out = self.trends(src_trend)
                # trend_out = self.revin(trend_out, 'denorm')

                temporal_out = seasonal_out + trend_out
                # x += trend_out
            return self.sublayer[1](temporal_out, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, src_trend=None):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            # x = layer(x)
            x = layer(x, src_trend)
        return self.norm(x)

def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    # np.triu(a, k)是取矩阵a的上三角数据，三角的斜线位置由k的值确定。
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   # 1 means reachable; 0 means unreachable

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
            x = self.sublayer[2](x, self.feed_forward_gcn)  # output:  (batch, N, T', d_model)
            return x
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        '''

        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size = 16, stride = 1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # padding on the both ends of time series
        front = x[:, :, :1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=2)  # torch.cat 是 PyTorch 中用于在指定的维度上连接张量

        moving_mean = self.avgpool(x)
        return moving_mean

class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size = 16):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = moving_avg(kernel_size = kernel_size)
        self.dec_layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        moving_mean = []
        mov_avg = self.moving_avg(x)
        moving_mean.append(mov_avg.unsqueeze(-1))

        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.dec_layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean

class EncoderDecoder(nn.Module):
    def __init__(self, series_dec, revin, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.series_dec = series_dec
        # self.revin = revin
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        # self.trend = trend
        self.to(DEVICE)

    def forward(self, src, trg, process = 'first stage'):
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''
        """
        batch_size, N, T_in, F_in = src.size()
        batch, _, T_out, F_out = trg.size()

        # 时间序列分解
        src_season, src_trend = self.series_dec(src.view(batch_size, N, -1))
        trg_season, trg_trend = self.series_dec(trg.view(batch, N, -1))

        src_season = src_season.view(batch_size, N, T_in, F_in)
        src_trend = src_trend.view(batch_size, N, T_in, F_in)
        trg_season = trg_season.view(batch, N, T_out, F_out)
        trg_trend = trg_trend.view(batch, N, T_out, F_out)
        """

        encoder_output = self.encode(src)
        # dec_output = self.decode(encoder_output, encoder_output)  # trg(4, 307, 12, 1), encoder_output(4, 307, 12, 1)
        dec_output = self.decode(trg, encoder_output) # trg(4, 307, 12, 1), encoder_output(4, 307, 12, 64)
        output = dec_output
        """
        if process == 'first stage':
            return encoder_output

        else:
            dec_output = self.decode(encoder_output, encoder_output)  # trg(4, 307, 12, 1), encoder_output(4, 307, 12, 1)
            # dec_output = self.decode(trg, encoder_output) # trg(4, 307, 12, 1), encoder_output(4, 307, 12, 64)
            output = dec_output
            return output
        """
        return output


    def encode(self, src):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        batch_size, N, T_in, F_in = src.size()

        # src = self.revin(src, 'norm')

        # 时间序列分解
        src_season, src_trend = self.series_dec(src.view(batch_size, N, -1))
        src_trend = src_trend.view(batch_size, N, T_in, F_in)

        # 再把src_trend放入src_embed中也做一下
        src_trend_emd = self.src_embed(src_trend)
        h = self.src_embed(src)
        enc_output = self.encoder(h, src_trend_emd)  # (batch_size, N, T_in, d_model)

        # pre_output = self.prediction_generator(enc_output) # (batch_size, N, T_in, F)

        # pre_output = self.revin(pre_output, 'denorm')
        return enc_output
        # return pre_output
        # return self.encoder(h)
        # return self.encoder(self.src_embed(src))

    def decode(self, trg, encoder_output):
        dec_output = self.decoder(self.trg_embed(trg), encoder_output)
        # dec_output = encoder_output
        pre_output = self.prediction_generator(dec_output)
        # output = encoder_output + pre_output
        return pre_output

def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, geo_mx, sem_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):

    c = copy.deepcopy  # 定义一个深拷贝

    # 这两种归一化方法都旨在处理图卷积操作中的数值稳定性和梯度传播问题。选择哪种归一化方式通常取决于具体的应用场景和实验效果。
    # sym_norm_Adj(adj_mx)：构造对称的归一化拉普拉斯矩阵，norm_Adj：构造归一化的拉普拉斯矩阵
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵
    norm_Geo_matrix = torch.from_numpy(norm_Adj(geo_mx)).type(torch.FloatTensor).to(DEVICE)
    norm_Sem_matrix = torch.from_numpy(norm_Adj(sem_mx)).type(torch.FloatTensor).to(DEVICE)

    num_of_vertices = norm_Adj_matrix.shape[0]

    src_dense = nn.Linear(encoder_input_size, d_model)  # encoder_input_size = 1, d_model = 64

    if ScaledSAt:  # employ spatial self attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(norm_Adj_matrix, norm_Geo_matrix, norm_Sem_matrix, d_model, d_model), dropout=dropout)
    else:  # 不带attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)

    trg_dense = nn.Linear(decoder_output_size, d_model)  # target input projection

    # encoder temporal position embedding
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict,
                  num_of_hours * num_for_predict)
    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7 * 24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    # hour = 1：h_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index

    print('TemporalPositionalEncoding max_len:', max_len)
    print('w_index:', w_index)
    print('d_index:', d_index)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    # employ traditional self attention
    attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout, domain='Fourier')
    attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
    # att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout, domain='ESM')
    att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and (TE):
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)

    trend = nn.Sequential(
        # nn.Linear(encoder_input_size, d_model),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        # nn.Linear(d_model, decoder_output_size)
        nn.Linear(d_model, d_model)
    )

    revin_trend = RevIN(encoder_input_size)

    encoderLayer = EncoderLayer(d_model, attn_ss, c(position_wise_gcn), dropout, trend,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    decoder = Decoder(decoderLayer, num_layers)

    generator = nn.Linear(d_model, decoder_output_size)

    series_dec = series_decomp_multi(kernel_size = 12)

    model = EncoderDecoder(series_dec,
                           revin_trend,
                           encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE)

    # param init
    for p in model.parameters():
        if p.dim() > 1:
            # 是一个服从均匀分布的Glorot初始化器，预防一些参数过大或过小的情况，再保证方差一样的情况下进行缩放，便于计算
            nn.init.xavier_uniform_(p)

    return model





















