import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
# --- 修改点1：导入旧版 Vocab 和 Python 标准计数器 ---因为我的torchtext版本为0.6.0所以高版本的导入方式不兼容。
from torchtext.vocab import Vocab
from collections import Counter
# ------------------------------------------------
from datasets import load_from_disk
import spacy
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DataManager:
    def __init__(self, batch_size=128, min_freq=2, max_len=100):
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.max_len = max_len

        # 1. 加载 Tokenizers
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            print("警告：未找到 Spacy 模型，请确保已下载。")
            raise

        # 2. 加载数据集 (本地)
        print("Loading Multi30k dataset from local disk...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        local_data_path = os.path.join(project_root, "data")

        if os.path.exists(local_data_path):
            print(f"  Found local data at: {local_data_path}")
            self.dataset = load_from_disk(local_data_path)
        else:
            raise FileNotFoundError(f"错误：未找到 data 文件夹：{local_data_path}")

        # 3. 构建词表
        print("Building Vocabularies (Legacy Mode)...")
        self.vocab_src = self.build_vocab(self.dataset['train'], 'de', self.spacy_de)
        self.vocab_trg = self.build_vocab(self.dataset['train'], 'en', self.spacy_en)

        # 获取特殊 Token 的索引
        # 在 torchtext 0.6.0 中，我们需要通过 stoi (string to index) 字典来获取
        self.PAD_IDX = self.vocab_src.stoi['<pad>']
        self.SOS_IDX = self.vocab_trg.stoi['<sos>']
        self.EOS_IDX = self.vocab_trg.stoi['<eos>']
        self.UNK_IDX = self.vocab_src.stoi['<unk>']  # 获取 unk 索引用于后续处理

        print(f"Source Vocab Size: {len(self.vocab_src)}")
        print(f"Target Vocab Size: {len(self.vocab_trg)}")

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    # --- 修改点2：重写 build_vocab 适配 0.6.0 ---
    def build_vocab(self, data, lang, spacy_model):
        print(f"  - Building vocab for {lang}...")
        counter = Counter()
        for example in data:
            tokens = [tok.text for tok in spacy_model.tokenizer(example[lang])]
            counter.update(tokens)

        # 直接使用 Vocab 类实例化
        # 0.6.0 的 Vocab 构造函数接受 Counter
        vocab = Vocab(counter, min_freq=self.min_freq, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
        return vocab

    # -----------------------------------------

    def data_process(self, raw_data_iter):
        data = []
        for raw_txt in raw_data_iter:
            # --- 修改点3：安全的 Token 转 Index ---
            # 旧版 Vocab 可能没有 set_default_index，所以我们手动处理 <unk>
            # .stoi.get(token, unk_index) 是最稳妥的写法

            src_tokens = self.tokenize_de(raw_txt['de'])
            src_indices = [self.vocab_src.stoi.get(token, self.vocab_src.stoi['<unk>']) for token in src_tokens]
            src_tensor = torch.tensor(src_indices, dtype=torch.long)

            trg_tokens = self.tokenize_en(raw_txt['en'])
            trg_indices = [self.vocab_trg.stoi.get(token, self.vocab_trg.stoi['<unk>']) for token in trg_tokens]
            trg_tensor = torch.tensor(trg_indices, dtype=torch.long)
            # -----------------------------------

            # 添加 <sos> 和 <eos>
            trg_tensor = torch.cat([torch.tensor([self.SOS_IDX]), trg_tensor, torch.tensor([self.EOS_IDX])])
            src_tensor = torch.cat([torch.tensor([self.SOS_IDX]), src_tensor, torch.tensor([self.EOS_IDX])])

            if len(src_tensor) <= self.max_len and len(trg_tensor) <= self.max_len:
                data.append((src_tensor, trg_tensor))
        return data

    def collate_fn(self, batch):
        src_batch, trg_batch = [], []
        for src_item, trg_item in batch:
            src_batch.append(src_item)
            trg_batch.append(trg_item)

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=self.PAD_IDX, batch_first=True)
        return src_batch, trg_batch

    def get_loaders(self):
        print("Processing data (tokenizing and converting to indices)...")
        train_data = self.data_process(self.dataset['train'])
        valid_data = self.data_process(self.dataset['validation'])
        test_data = self.data_process(self.dataset['test'])

        # num_workers=0 在 Windows 上通常更稳定，避免多进程报错
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,
                                  num_workers=0)
        valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
                                  num_workers=0)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
                                 num_workers=0)

        return train_loader, valid_loader, test_loader
