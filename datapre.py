import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_qwen4')
NODE_FEATURE_PATH = os.path.join(DATA_DIR, 'node_features_qwen_4B.npy')
NODE_INDEX_PATH = os.path.join(DATA_DIR, 'node_index.json')
GRAPH_PATH = os.path.join(DATA_DIR, 'graph.csv')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'label_map.csv')
OUT_PATH = os.path.join(DATA_DIR, 'pyg_graph.pt')

# 1. 加载与对齐
def load_data():
    features = np.load(NODE_FEATURE_PATH)
    with open(NODE_INDEX_PATH, 'r', encoding='utf-8') as f:
        node_index = json.load(f)
    id2idx = {nid: i for i, nid in enumerate(node_index)}
    graph_df = pd.read_csv(GRAPH_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return features, id2idx, graph_df, train_df, test_df

# 2. 格式化数据
def build_edge_index(graph_df, id2idx):
    # 用id2idx映射，得到整数索引
    # 将整数转换为字符串进行映射
    src = graph_df['src'].astype(str).map(id2idx)
    dst = graph_df['dst'].astype(str).map(id2idx)
    # 过滤掉未能映射的边
    mask = src.notna() & dst.notna()
    src = src[mask].astype(int)
    dst = dst[mask].astype(int)
    # 高效拼接
    edge_index_np = np.vstack([src.values, dst.values])
    edge_index = torch.from_numpy(edge_index_np).long()
    return edge_index

def build_label_vector(train_df, id2idx, num_nodes, label_map_path=LABEL_MAP_PATH):
    label_vec = torch.full((num_nodes,), -1, dtype=torch.long)
    # 若有label_map则做映射（label_name->id）
    label2id = None
    if os.path.exists(label_map_path):
        label_map = pd.read_csv(label_map_path)
        if 'id' in label_map.columns and 'label_name' in label_map.columns:
            label2id = {row['label_name']: row['id'] for _, row in label_map.iterrows()}
        else:
            raise ValueError('label_map.csv需包含id和label_name两列')
    for _, row in train_df.iterrows():
        idx = id2idx.get(str(row['id']))
        if idx is not None:
            label = row['label']
            # 如果label是字符串且有label2id映射，则查映射，否则直接用
            if label2id and isinstance(label, str):
                label = label2id[label]
            label_vec[idx] = int(label)
    return label_vec

def build_masks(train_df, test_df, id2idx, num_nodes):
    """
    构建训练和测试掩码（无验证集）
    """
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # 所有训练集节点都用于训练
    for _, row in train_df.iterrows():
        idx = id2idx.get(str(row['id']))
        if idx is not None:
            train_mask[idx] = True
    # 测试集节点
    for _, row in test_df.iterrows():
        idx = id2idx.get(str(row['id']))
        if idx is not None:
            test_mask[idx] = True
    return train_mask, test_mask

def main():
    features, id2idx, graph_df, train_df, test_df = load_data()
    num_nodes = len(id2idx)
    x = torch.tensor(features, dtype=torch.float)
    edge_index = build_edge_index(graph_df, id2idx)
    y = build_label_vector(train_df, id2idx, num_nodes)
    train_mask, test_mask = build_masks(train_df, test_df, id2idx, num_nodes)
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    torch.save(data, OUT_PATH)
    print(f"Saved PyG Data object to {OUT_PATH}")
    print(f"Training nodes: {train_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")

if __name__ == '__main__':
    main()
