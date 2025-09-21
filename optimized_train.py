#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版训练脚本 - 包含多种优化策略
"""

import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import math

# 设置设备
if torch.cuda.is_available():
    device = torch.device('cuda')
    use_gpu = 'GPU'
    print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    use_gpu = 'CPU'
    print("使用CPU训练")

# 加载词汇表
try:
    dict_datas = json.load(open('../data/dict_datas.json', 'r'))
    word2id = dict_datas["word2id"]
    id2word = dict_datas["id2word"]
    print(f"词汇表大小: {len(word2id)}")
except FileNotFoundError:
    print("无法找到词汇表文件'dict_datas.json'，请先运行process_data.py")
    exit()

# 从gpt_model导入模型
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gpt_model import GPT

# 优化后的训练参数
class TrainingConfig:
    def __init__(self):
        # 基础参数
        self.batch_size = 16  # 增加batch size
        self.epochs = 10      # 增加训练轮数
        self.learning_rate = 5e-4  # 学习率
        self.clip = 1.0       # 梯度裁剪
        
        # 优化参数
        self.warmup_steps = 1000  # 学习率预热步数
        self.weight_decay = 0.01  # 权重衰减
        self.dropout = 0.1        # Dropout率
        
        # 调度器参数
        self.scheduler_type = 'cosine'  # 学习率调度器类型
        
        # 早停参数
        self.patience = 3  # 早停耐心值
        self.min_delta = 0.001  # 最小改善阈值

config = TrainingConfig()
print(f"训练配置: batch_size={config.batch_size}, epochs={config.epochs}, lr={config.learning_rate}")

# 加载数据
try:
    with open('../data/dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()
    print(f"加载了 {len(datas)} 条训练数据")
except FileNotFoundError:
    print("无法找到数据集文件'dataset.txt'，请先运行process_data.py")
    exit()

# 数据增强函数
def augment_data(datas):
    """数据增强：添加噪声、同义词替换等"""
    augmented = []
    for data in datas:
        augmented.append(data)
        # 可以在这里添加数据增强逻辑
        # 例如：同义词替换、随机插入等
    return augmented

# 处理数据
def make_data(datas):
    train_datas = []
    for data in datas:
        data = data.strip()
        train_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>']
        train_datas.append(train_data)
    return train_datas

# 数据增强
augmented_datas = augment_data(datas)
train_data = make_data(augmented_datas)
train_num_data = [[word2id.get(word, word2id["<unk>"]) for word in line] for line in train_data]

# 创建数据集（与之前相同）
class MyDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1]
        decoder_output = data[1:]
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        return {
            "decoder_input": decoder_input,
            "decoder_input_len": decoder_input_len,
            "decoder_output": decoder_output,
            "decoder_output_len": decoder_output_len
        }

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)
        
        for d in batch:
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        return decoder_inputs, decoder_outputs

# 创建数据加载器
dataset = MyDataSet(train_num_data)
data_loader = Data.DataLoader(dataset, batch_size=config.batch_size, collate_fn=dataset.padding_batch, shuffle=True)

# 创建模型
model = GPT().to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 学习率调度器
def get_lr_scheduler(optimizer, config):
    if config.scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    else:
        return None

# 早停类
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# 优化后的训练函数
def train_step(model, data_loader, optimizer, criterion, scheduler, config, print_every=10):
    model.train()
    epoch_loss = 0
    total_steps = len(data_loader)
    
    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader, desc="训练中")):
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        
        optimizer.step()
        
        # 学习率预热
        if i < config.warmup_steps:
            lr = config.learning_rate * (i + 1) / config.warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        epoch_loss += loss.item()
        
        if (i + 1) % print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\t批次 {i+1}/{total_steps}, 损失: {loss.item():.4f}, 学习率: {current_lr:.6f}')
    
    return epoch_loss / len(data_loader)

# 验证函数
def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for dec_inputs, dec_outputs in data_loader:
            dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
            outputs, _ = model(dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# 开始训练
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = get_lr_scheduler(optimizer, config)
early_stopping = EarlyStopping(config.patience, config.min_delta)

loss_values = []
val_loss_values = []

print("\n开始优化训练...")
for epoch in range(config.epochs):
    start_time = time.time()
    
    # 训练
    train_loss = train_step(model, data_loader, optimizer, criterion, scheduler, config, print_every=20)
    loss_values.append(train_loss)
    
    # 验证（这里用训练集的一部分作为验证集）
    val_loss = validate(model, data_loader, criterion)
    val_loss_values.append(val_loss)
    
    # 学习率调度
    if scheduler:
        scheduler.step()
    
    end_time = time.time()
    epoch_mins, epoch_secs = int((end_time - start_time) / 60), int((end_time - start_time) % 60)
    
    print(f'Epoch: {epoch + 1:02} | 时间: {epoch_mins}m {epoch_secs}s')
    print(f'训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}')
    
    # 保存最佳模型
    if epoch == 0 or val_loss < min(val_loss_values[:-1]):
        torch.save(model.state_dict(), '../models/GPT2_optimized.pt')
        print(f'保存最佳模型: GPT2_optimized.pt')
    
    # 早停检查
    if early_stopping(val_loss):
        print(f"早停触发，在第 {epoch + 1} 轮停止训练")
        break
    
    print('-' * 50)

# 绘制损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_values, label='训练损失', color='blue')
plt.plot(val_loss_values, label='验证损失', color='red')
plt.title('训练和验证损失曲线')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_values, label='训练损失', color='blue')
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../docs/optimized_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n优化训练完成！")
print(f"最佳模型已保存为: GPT2_optimized.pt")
print(f"损失曲线已保存为: optimized_loss_curve.png")
print(f"最终训练损失: {loss_values[-1]:.4f}")
print(f"最终验证损失: {val_loss_values[-1]:.4f}")
