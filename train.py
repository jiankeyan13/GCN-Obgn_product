import torch
import torch.nn.functional as F
import os
import numpy as np
import random
from torch.optim import AdamW
from gat_model import GAT
import pandas as pd

# 设置随机种子
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(os.path.dirname(__file__), 'data_test', 'pyg_graph.pt')
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

# 训练函数
def train_and_eval(gat_model, gat_optimizer, gat_scheduler, criterion, epochs):
    best_avg_acc = 0
    best_gat_state = None
    best_epoch = 0
    acc_history = []  # 记录所有epoch的准确率
    
    for epoch in range(1, epochs + 1):
        # 训练GAT
        gat_model.train()
        gat_optimizer.zero_grad()
        gat_out = gat_model(data.x, data.edge_index)
        train_loss = criterion(gat_out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        gat_optimizer.step()
        gat_scheduler.step()
        
        # 验证集预测
        gat_model.eval()
        with torch.no_grad():
            gat_logits = gat_model(data.x, data.edge_index)
            val_mask = data.val_mask
            gat_probs = F.softmax(gat_logits[val_mask], dim=1)
            gat_pred = gat_probs.argmax(dim=1)
            true_label = data.y[val_mask]
            gat_acc = (gat_pred == true_label).float().mean().item()
            val_loss = criterion(gat_logits[val_mask], data.y[val_mask])
        
        # 记录准确率历史
        acc_history.append(gat_acc)
        
        # 只用已完成的epoch计算平均（[-2, -1, 0]）
        if epoch >= 3:
            avg_acc = sum(acc_history[epoch-3:epoch]) / 3
        else:
            avg_acc = sum(acc_history[:epoch]) / epoch
        
        # 记录最佳平均准确率
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_gat_state = gat_model.state_dict()
            best_epoch = epoch
        
        # 打印（显示当前准确率、训练loss、验证loss）
        if epoch % 5 == 0:
            print(f"[Epoch {epoch:03d}] GAT Val Acc: {gat_acc:.4f} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    # 恢复最佳参数
    gat_model.load_state_dict(best_gat_state)
    return best_avg_acc, best_epoch

# 初始化模型和优化器
print("\n--- Training GAT (with validation set) ---")
gat_model = GAT(in_channels, gat_hidden, out_channels, heads=gat_heads, dropout=gat_dropout).to(device)
gat_optimizer = AdamW(gat_model.parameters(), lr=gat_lr, weight_decay=gat_weight_decay)
gat_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gat_optimizer, T_max=gat_epochs)
criterion = torch.nn.CrossEntropyLoss()

best_avg_acc, best_epoch = train_and_eval(
    gat_model, gat_optimizer, gat_scheduler, criterion, gat_epochs
)

print(f"\nGAT验证集最佳平均准确率（[-2,-1,0]窗口）: {best_avg_acc:.4f}")
print(f"最佳平均准确率对应的epoch: {best_epoch}")
