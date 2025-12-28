import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 标准正弦/余弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1. Linear Projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. Split Heads: [batch, len, d_model] -> [batch, len, n_head, d_k] -> [batch, n_head, len, d_k]
        Q = Q.view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention (Using Einsum for Graduate Level Style)
        # energy: [batch, n_head, query_len, key_len]
        energy = torch.einsum("bhqd, bhkd -> bhqk", Q, K) / self.scale.to(Q.device)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)  # 保存这个 attention map 以后可以可视化

        # 4. Weighted Sum
        # out: [batch, n_head, query_len, d_k]
        out = torch.einsum("bhqk, bhvd -> bhqd", attention, V)

        # 5. Concat
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        out = self.fc_out(out)
        return out, attention


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch, seq_len, d_model]
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, pre_norm=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.pff = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm  # 实验 2.1 关键变量

    def forward(self, src, src_mask):
        # Pre-Norm: Norm -> Sublayer -> Add
        # Post-Norm: Sublayer -> Add -> Norm

        if self.pre_norm:
            # Self Attention
            _src, _ = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), src_mask)
            src = src + self.dropout(_src)
            # PFF
            _src = self.pff(self.norm2(src))
            src = src + self.dropout(_src)
        else:
            # Self Attention
            _src, _ = self.self_attn(src, src, src, src_mask)
            src = self.norm1(src + self.dropout(_src))
            # PFF
            _src = self.pff(src)
            src = self.norm2(src + self.dropout(_src))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, pre_norm=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.pff = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # enc_src: Encoder output

        if self.pre_norm:
            # 1. Self Attention (Decoder-Decoder)
            _trg, _ = self.self_attn(self.norm1(trg), self.norm1(trg), self.norm1(trg), trg_mask)
            trg = trg + self.dropout(_trg)

            # 2. Encoder Attention (Decoder-Encoder)
            # Query from Decoder, Key/Value from Encoder
            _trg, attention_weights = self.enc_attn(self.norm2(trg), enc_src, enc_src, src_mask)
            trg = trg + self.dropout(_trg)

            # 3. PFF
            _trg = self.pff(self.norm3(trg))
            trg = trg + self.dropout(_trg)

        else:
            _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
            trg = self.norm1(trg + self.dropout(_trg))

            _trg, attention_weights = self.enc_attn(trg, enc_src, enc_src, src_mask)
            trg = self.norm2(trg + self.dropout(_trg))

            _trg = self.pff(trg)
            trg = self.norm3(trg + self.dropout(_trg))

        return trg, attention_weights


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n_head, n_layers, d_ff, max_len, dropout, device, pre_norm=False):
        super().__init__()
        self.device = device

        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, d_ff, dropout, pre_norm) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_ff, dropout, pre_norm) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, trg_vocab)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        self.pre_norm = pre_norm

        # 终极 LayerNorm (Pre-Norm 架构需要在最后加一个 Norm)
        if pre_norm:
            self.final_norm_enc = nn.LayerNorm(d_model)
            self.final_norm_dec = nn.LayerNorm(d_model)

    def make_src_mask(self, src, pad_idx):
        # src: [batch, len]
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # mask: [batch, 1, 1, len]
        return mask

    def make_trg_mask(self, trg, pad_idx):
        # trg: [batch, len]
        trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, len]
        trg_len = trg.shape[1]

        # Causal Mask (Lower Triangular)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        mask = trg_pad_mask & trg_sub_mask
        return mask

    def forward(self, src, trg):
        # src: [batch, src_len]
        # trg: [batch, trg_len]

        src_mask = self.make_src_mask(src, 1)  # 假设 PAD_IDX = 1，实际从 config 读
        trg_mask = self.make_trg_mask(trg, 1)

        # Encoder
        src = self.dropout((self.src_embedding(src) * self.scale) + self.pos_encoding(
            self.src_embedding(src)))  # 这里通常不用 pos_enc(embedded), 而是 pos_enc(zeros) 叠加，上面PosEnc写好了
        # 修正调用:
        src = self.src_embedding(src) * self.scale
        src = self.pos_encoding(src)
        src = self.dropout(src)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        if self.pre_norm:
            src = self.final_norm_enc(src)

        # Decoder
        trg = self.trg_embedding(trg) * self.scale
        trg = self.pos_encoding(trg)
        trg = self.dropout(trg)

        for layer in self.decoder_layers:
            trg, attention = layer(trg, src, trg_mask, src_mask)

        if self.pre_norm:
            trg = self.final_norm_dec(trg)

        output = self.fc_out(trg)
        return output, attention
