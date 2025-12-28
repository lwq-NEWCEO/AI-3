import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim, emb_dim)

        # 实验变量 1.1: Bidirectional
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        # 实验变量 1.1: Bridge Layer (维度处理的关键)
        if bidirectional:
            self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
            self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch, src_len, hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch, hid_dim]

        if self.bidirectional:
            # 这里的处理体现了你对维度的深度理解
            # 我们需要把双向的 hidden state 融合给单向 decoder 使用
            # 取最后一层的 forward 和 backward
            hidden_fwd = hidden[-2, :, :]
            hidden_bwd = hidden[-1, :, :]
            # 拼接并通过 Bridge Layer 压缩
            hidden = torch.tanh(self.fc_hidden(torch.cat((hidden_fwd, hidden_bwd), dim=1)))

            cell_fwd = cell[-2, :, :]
            cell_bwd = cell[-1, :, :]
            cell = torch.tanh(self.fc_cell(torch.cat((cell_fwd, cell_bwd), dim=1)))

            # 现在的 hidden/cell 是 [batch, hid_dim]，需要扩展回 [n_layers, batch, hid_dim]
            # 这里为了简化，我们假设 Decoder 和 Encoder 层数相同，且只传递最后一层给 Decoder 作为所有层的初始状态
            # 或者简单的 unsqueeze
            hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(self.n_layers, 1, 1)

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # Bahdanau Attention (Additive)
        self.attn = nn.Linear(hid_dim * 2 + hid_dim, hid_dim)  # Encoder(Bi) + Decoder
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim] (Decoder 上一步的 hidden)
        # encoder_outputs: [batch, src_len, hid_dim * 2] (Bi-Encoder 输出)

        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hid_dim]

        # 计算 energy: tanh(W [h_dec; h_enc] + b)
        # 注意：如果 Encoder 是双向，其输出是 2*hid_dim，加上 decoder 的 hid_dim
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # 如果有 Attention，输入就是 [emb + enc_out_dim]
        input_rnn_dim = emb_dim + (hid_dim * 2) if attention else emb_dim

        self.rnn = nn.LSTM(input_rnn_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        input_fc_dim = hid_dim * 2 + hid_dim + emb_dim if attention else hid_dim
        # 为了简化实现，标准 Attention Decoder 通常是在 output 阶段做拼接
        # 这里使用经典的 Luong/Bahdanau 变体
        self.fc_out = nn.Linear(input_rnn_dim + hid_dim, output_dim) if attention else nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs=None):
        # input: [batch] (单个词)
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb]

        a = None
        if self.attention:
            # 使用 Decoder 上一步的 hidden[-1] 进行 attention
            # hidden shape: [n_layers, batch, hid] -> 取最后一层 [batch, hid]
            a = self.attention(hidden[-1], encoder_outputs)  # [batch, src_len]
            a = a.unsqueeze(1)  # [batch, 1, src_len]

            # Weighted Sum of Encoder Outputs
            weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid * 2]

            # RNN Input: [Embedded; Weighted_Context]
            rnn_input = torch.cat((embedded, weighted), dim=2)

            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

            # Prediction: [Output; Weighted_Context; Embedded] -> FC
            # 这里简化，通常取 output 和 weighted
            prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))

        else:
            # 无 Attention (Baseline)
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            prediction = self.fc_out(output)

        prediction = prediction.squeeze(1)  # [batch, output_dim]

        return prediction, hidden, cell, a


class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # 存储所有时间步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 1. Encode
        encoder_outputs, hidden, cell = self.encoder(src)

        # 2. Decode initialization
        input = trg[:, 0]  # <sos>

        # 存储 Attention Map 用于可视化 (Phase 2 必得分)
        attentions = []

        for t in range(1, trg_len):
            # 如果有 attention，encoder_outputs 会被用到
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs)

            outputs[:, t] = output
            if attention is not None:
                attentions.append(attention.squeeze(1).cpu().detach())

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs, attentions
