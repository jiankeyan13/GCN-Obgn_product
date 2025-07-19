import torch
import torch.nn.functional as F
import os
import numpy as np
import random
import pandas as pd
from torch.optim import AdamW
from gat_model import GAT

# 设置随机种子
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(os.path.dirname(__file__), 'data_qwen4', 'pyg_graph.pt')
data = torch.load(data_path, map_location=device, weights_only=False)
data = data.to(device)

# 参数
gat_hidden = 128
gat_heads = 4
gat_dropout = 0.15
gat_lr = 0.003
gat_weight_decay = 4e-4
gat_epochs = 150
in_channels = data.x.shape[1]
out_channels = int(data.y.max().item()) + 1

# 训练函数（train loss早停）
def train_gat_earlystop(model, optimizer, scheduler, criterion, epochs, patience=3, min_delta=1e-3):
    best_state = None
    best_epoch = 0
    stop_epoch = epochs
    train_loss_history = []
    patience_count = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_history.append(train_loss.item())
        # 打印
        if epoch % 5 == 0:
            print(f"[Epoch {epoch:03d}] Train Loss: {train_loss.item():.4f}")
        # 早停判断
        if epoch >= 4:
            recent_losses = train_loss_history[-3:]
            max_loss = max(recent_losses)
            min_loss = min(recent_losses)
            if max_loss - min_loss <= min_delta:
                print(f"Early stopping at epoch {epoch} (train loss change <= {min_delta} for last 3 epochs)")
                stop_epoch = epoch
                break
    return model, stop_epoch

print("\n--- Training GAT (Full Training Set, Train Loss Early Stopping) ---")
gat_model = GAT(in_channels, gat_hidden, out_channels, heads=gat_heads, dropout=gat_dropout).to(device)
gat_optimizer = AdamW(gat_model.parameters(), lr=gat_lr, weight_decay=gat_weight_decay)
gat_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gat_optimizer, T_max=gat_epochs)
criterion = torch.nn.CrossEntropyLoss()

gat_model, stop_epoch = train_gat_earlystop(gat_model, gat_optimizer, gat_scheduler, criterion, gat_epochs, patience=3, min_delta=1e-3)
print(f"训练在第{stop_epoch}个epoch停止（或达到最大轮数）")

# 预测test集
print("\n--- Predicting Test Set ---")
gat_model.eval()
with torch.no_grad():
    logits = gat_model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

# 读取test.csv，保证顺序
cur_dir = os.path.dirname(__file__)
test_csv_path = os.path.join(cur_dir, 'data_qwen4', 'test.csv')
test_df = pd.read_csv(test_csv_path)

# 加载节点索引映射
with open(os.path.join(cur_dir, 'data_qwen4', 'node_index.json'), 'r', encoding='utf-8') as f:
    node_index = f.read()
    node_index = eval(node_index) if isinstance(node_index, str) else node_index
id2idx = {nid: i for i, nid in enumerate(node_index)}

# 生成提交文件，顺序与test.csv一致
submission = []
for test_id in test_df['id']:
    idx = id2idx.get(str(test_id))
    if idx is not None and data.test_mask[idx]:
        submission.append({'id': test_id, 'outcome': int(pred[idx].item())})
    else:
        submission.append({'id': test_id, 'outcome': 0})  # 若找不到，默认类别0

sub_df = pd.DataFrame(submission)
sub_df.to_csv('output_task2.csv', index=False, columns=['id', 'outcome'])
print('output_task2.csv saved!') 