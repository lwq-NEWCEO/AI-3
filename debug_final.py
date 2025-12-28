import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils import DataManager
from models.transformer_model import Seq2SeqTransformer
from evaluate_transformer import translate_sentence


def diagnose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dm = DataManager(batch_size=1)
    _, _, test_loader = dm.get_loaders()

    # 加载 Final Fix 模型
    INPUT_DIM = len(dm.vocab_src)
    OUTPUT_DIM = len(dm.vocab_trg)
    HID_DIM = 256
    model = Seq2SeqTransformer(INPUT_DIM, OUTPUT_DIM, HID_DIM, 8, 2, 256, 200, 0.1, device, pre_norm=True).to(device)
    model.load_state_dict(torch.load('output/trans_final_fix/model.pt', map_location=device))

    print("\n====== 💀 尸检报告: Final Fix 模型输出 ======")
    count = 0
    for src, trg in test_loader:
        if count >= 5: break

        pred_indices = translate_sentence(src[0], model, device, sos_idx=dm.vocab_trg.stoi['<sos>'],
                                          eos_idx=dm.vocab_trg.stoi['<eos>'])

        src_text = " ".join([dm.vocab_src.itos[idx] for idx in src[0].numpy() if idx not in [1, 2, 3]])
        ref_text = " ".join([dm.vocab_trg.itos[idx] for idx in trg[0].numpy() if idx not in [1, 2, 3]])
        pred_text = " ".join([dm.vocab_trg.itos[idx] for idx in pred_indices if idx != 3])

        print(f"源文: {src_text}")
        print(f"参考: {ref_text}")
        print(f"预测: {pred_text}")  # <--- 重点看这里！
        print("-" * 30)
        count += 1


if __name__ == "__main__":
    diagnose()
