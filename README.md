# GPT-2 中文对话模型实现与训练

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**课程作业项目** - 基于PyTorch从零实现GPT-2模型，并训练中文对话系统。

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+ (推荐3.9或3.10)
- **PyTorch**: 2.0+ (支持CUDA 11.8+)
- **内存**: 8GB+ RAM (推荐16GB+)
- **GPU**: NVIDIA GPU (可选，用于加速训练)

### 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd GPT-2-Model-Implementation

# 安装依赖包
pip install -r requirements.txt
```

### 运行步骤

#### 方法一：一键运行
```bash
python scripts/quick_start.py
```

#### 方法二：分步执行
1. **数据预处理**
```bash
python scripts/process_data.py
```

2. **模型训练**
```bash
python scripts/optimized_train.py
```

3. **模型测试**
```bash
python scripts/demo.py
```

## 📊 实验结果

### 模型性能指标
- **模型大小**: 159.69 MB
- **参数数量**: 41,860,830 个
- **词汇表大小**: 1,758 个字符
- **训练数据**: 3,339 条对话

### 训练性能
- **训练时间**: ~5分钟 (RTX 4060 Laptop GPU)
- **推理速度**: ~4ms (单次生成)
- **内存占用**: ~2GB (训练时)
- **最终损失**: 1.10 (5个epochs)

### 模型架构参数
- **层数**: 12层Transformer解码器
- **注意力头数**: 12个
- **隐藏维度**: 768
- **最大序列长度**: 1024

## 📈 训练配置

### 超参数设置
- **Batch Size**: 16
- **Learning Rate**: 5e-4
- **Epochs**: 10
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0

## 🎯 代码示例

### 模型加载和推理
```python
from src.gpt_model import GPT
import torch
import json

# 加载词汇表
with open('data/dict_datas.json', 'r', encoding='utf-8') as f:
    dict_datas = json.load(f)
word2id = dict_datas["word2id"]
id2word = dict_datas["id2word"]

# 加载模型
model = GPT()
model.load_state_dict(torch.load('models/GPT2.pt'))
model.eval()

# 生成回复
def generate_response(input_text, max_length=50):
    input_ids = [word2id.get(char, word2id["<unk>"]) for char in input_text]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs, _ = model(input_tensor)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(1)], dim=1)
            
            if next_token.item() == word2id["<sep>"]:
                break
    
    # 转换回文本
    generated_ids = input_tensor[0].cpu().tolist()
    generated_text = ''.join([id2word[id] for id in generated_ids[len(input_ids):] 
                             if id != word2id["<pad>"]])
    
    return generated_text.replace('<sep>', '').strip()

# 使用示例
response = generate_response("你好")
print(f"回复: {response}")
```

## 📝 数据格式说明

### 训练数据格式
本项目使用问答对格式的中文对话数据：

```
你好
你好，很高兴见到你！

你叫什么名字？
我叫AI助手，很高兴为您服务。

你能做什么？
我可以回答各种问题，帮助您解决疑惑。
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenAI](https://openai.com/) - GPT-2原始论文和架构设计
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [ViudiraTech](https://github.com/ViudiraTech) - 项目基础代码
- 课程老师和同学们的指导与帮助

📝 **作业说明**: 本项目为课程作业，完整实现了GPT-2模型架构并训练了中文对话系统。
