# GPT-2 中文对话模型实现与训练

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**课程作业项目** - 基于PyTorch从零实现GPT-2模型，并训练中文对话系统。本项目包含完整的模型架构实现、数据处理、训练流程和性能优化。

## 📚 作业要求实现

### 核心功能
- ✅ **GPT-2模型架构**: 从零实现Transformer解码器架构
- ✅ **自注意力机制**: 多头自注意力机制实现
- ✅ **位置编码**: 可学习的位置嵌入
- ✅ **前馈网络**: 两层MLP结构
- ✅ **中文分词**: 字符级分词和词汇表构建

### 训练功能
- ✅ **数据预处理**: 对话数据格式化和清洗
- ✅ **训练循环**: 完整的训练流程实现
- ✅ **损失函数**: 交叉熵损失和梯度裁剪
- ✅ **优化器**: Adam优化器配置
- ✅ **模型保存**: 训练检查点和模型持久化

### 高级功能
- 🚀 **GPU加速**: CUDA支持，提升训练效率
- 📊 **数据增强**: 同义词替换、句子重排等技术
- 🔧 **模型优化**: 学习率调度、早停机制
- 📈 **性能监控**: 实时损失曲线和训练指标
- 🎯 **交互测试**: 命令行对话界面

## 📁 项目结构

```
GPT-2-Model-Implementation/
├── src/                           # 核心源代码
│   └── gpt_model.py              # GPT-2模型架构实现
├── scripts/                      # 训练和测试脚本
│   ├── process_data.py           # 数据预处理和词汇表构建
│   ├── optimized_train.py        # 模型训练脚本
│   ├── demo.py                   # 模型测试和对话界面
│   ├── data_augmentation.py      # 数据增强工具
│   ├── model_optimizer.py        # 模型性能分析和优化
│   └── quick_start.py            # 一键运行脚本
├── data/                         # 数据文件
│   ├── data.txt                  # 原始中文对话数据
│   ├── dataset.txt               # 预处理后的训练数据
│   ├── dict_datas.json           # 字符级词汇表
│   └── high_quality_data.txt     # 高质量训练数据
├── models/                       # 模型文件
│   └── GPT2.pt                   # 训练好的模型权重
├── docs/                         # 文档和结果
│   ├── loss_curve.png            # 训练损失曲线图
│   └── USAGE.md                  # 详细使用说明
├── requirements.txt              # Python依赖包
├── setup.py                      # 项目安装配置
├── .gitignore                    # Git版本控制忽略文件
├── LICENSE                       # MIT开源许可证
└── README.md                     # 项目说明文档
```

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

## 🔧 技术实现细节

### GPT-2模型架构
本项目实现了完整的GPT-2模型架构，包括：

- **多头自注意力机制**: 实现缩放点积注意力
- **位置编码**: 可学习的位置嵌入
- **前馈网络**: 两层MLP结构 (768→3072→768)
- **层归一化**: 在每个子层前进行归一化
- **残差连接**: 防止梯度消失问题

### 训练技术
- **学习率调度**: 余弦退火调度器
- **梯度裁剪**: 防止梯度爆炸
- **权重衰减**: L2正则化
- **早停机制**: 防止过拟合
- **数据增强**: 同义词替换、句子重排

### 性能优化
```bash
# 数据增强
python scripts/data_augmentation.py

# 模型性能分析
python scripts/model_optimizer.py
```

## 📈 训练配置

### 超参数设置
- **Batch Size**: 16
- **Learning Rate**: 5e-4
- **Epochs**: 10
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0

### 训练策略
- **学习率预热**: 1000步预热
- **余弦退火调度**: 学习率衰减
- **梯度裁剪**: 防止梯度爆炸
- **权重衰减**: L2正则化
- **早停机制**: 3轮无改善则停止

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

### 数据预处理流程
1. **文本清洗**: 去除特殊字符和多余空格
2. **字符级分词**: 将中文文本分解为字符序列
3. **词汇表构建**: 统计字符频率，构建字符到ID的映射
4. **序列填充**: 统一序列长度，添加padding token
5. **数据增强**: 同义词替换、句子重排等技术

## 📚 学习收获

通过完成这个GPT-2实现项目，我学到了：

### 技术知识
- **Transformer架构**: 深入理解自注意力机制和位置编码
- **PyTorch框架**: 熟练使用PyTorch进行深度学习开发
- **模型训练**: 掌握完整的训练流程和优化技巧
- **中文NLP**: 了解中文文本处理和字符级分词

### 工程实践
- **代码组织**: 模块化设计和项目结构规划
- **性能优化**: GPU加速、内存管理和训练效率提升
- **实验管理**: 超参数调优和结果分析
- **文档编写**: 技术文档和用户指南的编写

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenAI](https://openai.com/) - GPT-2原始论文和架构设计
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [ViudiraTech](https://github.com/ViudiraTech) - 项目基础代码
- 课程老师和同学们的指导与帮助

## 📞 联系方式

- **学生姓名**: [您的姓名]
- **学号**: [您的学号]
- **课程**: [课程名称]
- **邮箱**: [您的邮箱]

---

📝 **作业说明**: 本项目为课程作业，完整实现了GPT-2模型架构并训练了中文对话系统。