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

        # --- 修改 1.1: LSTM -> GRU ---
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        # Bridge Layer (GRU不需要处理cell state)
        if bidirectional:
            self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]

        # --- 修改 1.2: GRU只返回 outputs 和 hidden ---
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch, src_len, hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch, hid_dim]

        if self.bidirectional:
            # 双向GRU的hidden融合 (只处理hidden)
            hidden_fwd = hidden[-2, :, :]
            hidden_bwd = hidden[-1, :, :]
            hidden = torch.tanh(self.fc_hidden(torch.cat((hidden_fwd, hidden_bwd), dim=1)))

            # 将融合后的hidden扩展回decoder需要的层数
            hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # --- 修改 1.3: 返回值中没有 cell ---
        return outputs, hidden


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
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        input_rnn_dim = emb_dim + (hid_dim * 2) if attention else emb_dim

        # --- 修改 2.1: LSTM -> GRU ---
        self.rnn = nn.GRU(input_rnn_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        input_fc_dim = hid_dim * 2 + hid_dim + emb_dim if attention else hid_dim
        self.fc_out = nn.Linear(input_rnn_dim + hid_dim, output_dim) if attention else nn.Linear(hid_dim, output_dim)

    # --- 修改 2.2: forward 函数只接受 hidden，不再有 cell ---
    def forward(self, input, hidden, encoder_outputs=None):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        a = None
        if self.attention:
            a = self.attention(hidden[-1], encoder_outputs)
            a = a.unsqueeze(1)
            weighted = torch.bmm(a, encoder_outputs)
            rnn_input = torch.cat((embedded, weighted), dim=2)

            # --- 修改 2.3: GRU的调用和返回 ---
            output, hidden = self.rnn(rnn_input, hidden)

            prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        else:
            # --- 修改 2.4: 无Attention的GRU调用和返回 ---
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc_out(output)

        prediction = prediction.squeeze(1)

        # --- 修改 2.5: 返回值中没有 cell ---
        return prediction, hidden, a


class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # --- 修改 3.1: Encoder现在只返回 outputs 和 hidden ---
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]
        attentions = []

        for t in range(1, trg_len):
            # --- 修改 3.2: Decoder的调用和返回都不再有 cell ---
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs)

            outputs[:, t] = output
            if attention is not None:
                attentions.append(attention.squeeze(1).cpu().detach())

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs, attentions
