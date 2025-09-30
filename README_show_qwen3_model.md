# Qwen3模型结构分析工具

这个工具使用transformer库直接加载并分析Qwen3模型的结构，提供直观的模型层信息、参数统计和KV缓存分析。

## 功能特点

- 🚀 直接使用transformer库加载实际的Qwen3模型（非模拟类）
- 📊 详细的模型配置信息展示
- 📈 模型参数统计与占比分析
- 🏗️ 清晰的模型层次结构可视化
- 🔍 每层详细信息展示
- 💾 KV缓存机制分析
- 🧪 简单的推理测试功能
- 💾 将模型信息保存到文本文件

## 环境要求

确保你已经在conda的nanovllm虚拟环境中，并安装了以下依赖：

```bash
pip install torch>=2.4.0 transformers>=4.51.0 prettytable matplotlib
```

## 使用方法

1. 确保你的Qwen3模型路径正确（默认为`/opt/llm/Qwen3-0.6B/`）

2. 在nanovllm虚拟环境中运行脚本：

```bash
cd /opt/llm/laihongquan/projects/nano_vllm_note
python show_qwen3_model.py
```

## 输出内容

脚本执行后会输出以下内容：

1. **模型加载信息**：显示模型路径和加载状态
2. **模型配置详情**：以表格形式展示关键配置参数
3. **参数统计**：各组件参数数量、占比和总参数量
4. **模型层次结构**：树状结构展示模型的各层组件
5. **每层详细信息**：展示前3层和后3层的详细参数
6. **KV缓存分析**：识别并分析注意力层中的KV缓存组件
7. **推理测试**：使用简单提示词验证模型功能

此外，脚本还会生成一个`qwen3_model_structure_info.txt`文件，包含完整的模型信息。

## 注意事项

- 模型加载可能需要一定时间，具体取决于模型大小和硬件配置
- 为了避免内存不足，脚本默认使用`torch.float16`和`low_cpu_mem_usage=True`参数
- 如果模型路径不是`/opt/llm/Qwen3-0.6B/`，请修改脚本中的`model_path`参数
- 脚本支持中文显示，已配置matplotlib中文字体

## 与原show_qwen3_structure.py的区别

- 原脚本使用模拟类展示模型结构，而本脚本直接加载实际的Qwen3模型
- 本脚本提供了更详细的参数统计和模型分析功能
- 本脚本包含推理测试功能，可以验证模型是否正常工作
- 本脚本提供了更友好的表格化输出和信息保存功能

## 故障排除

如果遇到模型加载失败的问题，请检查：

1. 模型路径是否正确
2. 是否在nanovllm虚拟环境中
3. 依赖库是否已正确安装
4. 硬件内存是否足够加载模型

如果遇到中文显示问题，可以尝试安装额外的中文字体包：

```bash
# Ubuntu/Debian系统
apt-get update && apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei

# CentOS/RHEL系统
yum install -y wqy-microhei-fonts wqy-zenhei-fonts
```