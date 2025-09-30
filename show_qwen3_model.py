#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用transformer库直接加载Qwen3模型并展示其详细结构
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from prettytable import PrettyTable
import numpy as np

# 设置中文字体支持（避免打印表格时出现乱码）
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class Qwen3ModelAnalyzer:
    def __init__(self, model_path="/opt/llm/Qwen3-0.6B/"):
        """初始化模型分析器"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.total_params = 0
        self.layer_info = []  # 存储每层信息
        
    def load_model(self):
        """加载Qwen3模型和分词器"""
        print(f"正在从 {self.model_path} 加载Qwen3模型...")
        try:
            # 使用transformers库直接加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",  # 自动分配到可用设备
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 低CPU内存使用模式
            )
            
            # 将模型设置为评估模式
            self.model.eval()
            
            print(f"模型加载成功！")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def analyze_model_structure(self):
        """分析模型结构"""
        if self.model is None:
            print("请先加载模型")
            return
        
        print("\n=== Qwen3模型结构分析 ===")
        print(f"模型名称: {self.model_path}")
        print(f"模型类型: {self.model.__class__.__name__}")
        print(f"设备: {next(self.model.parameters()).device}")
        
        # 获取模型配置
        config = self.model.config
        self.print_config_details(config)
        
        # 分析参数统计
        self.analyze_parameters()
        
        # 打印模型层次结构
        self.print_model_hierarchy()
        
        # 打印每层的详细信息
        self.print_layer_details()
        
        # 分析KV缓存相关组件
        self.analyze_kv_cache()
        
        # 保存关键信息到文本文件
        self.save_model_info_to_file()
    
    def print_config_details(self, config):
        """打印模型配置详情"""
        print("\n=== 模型配置详情 ===")
        config_table = PrettyTable()
        config_table.field_names = ["配置项", "值"]
        
        # 收集关键配置信息
        config_items = [
            ("vocab_size", getattr(config, "vocab_size", "N/A")),
            ("hidden_size", getattr(config, "hidden_size", "N/A")),
            ("num_hidden_layers", getattr(config, "num_hidden_layers", "N/A")),
            ("num_attention_heads", getattr(config, "num_attention_heads", "N/A")),
            ("num_key_value_heads", getattr(config, "num_key_value_heads", "N/A")),
            ("intermediate_size", getattr(config, "intermediate_size", "N/A")),
            ("max_position_embeddings", getattr(config, "max_position_embeddings", "N/A")),
            ("rms_norm_eps", getattr(config, "rms_norm_eps", "N/A")),
            ("rope_theta", getattr(config, "rope_theta", "N/A")),
            ("torch_dtype", getattr(config, "torch_dtype", "N/A"))
        ]
        
        for name, value in config_items:
            config_table.add_row([name, value])
        
        print(config_table)
    
    def analyze_parameters(self):
        """分析模型参数统计"""
        print("\n=== 模型参数统计 ===")
        param_table = PrettyTable()
        param_table.field_names = ["组件", "参数数量", "占比"]
        
        total_params = 0
        param_counts = {}
        
        # 遍历所有参数
        for name, param in self.model.named_parameters():
            # 计算参数数量
            param_count = param.numel()
            total_params += param_count
            
            # 提取组件名称（按照点分割取第一部分）
            components = name.split(".")
            if len(components) > 1:
                component_name = components[0]
            else:
                component_name = name
            
            # 更新组件参数计数
            if component_name not in param_counts:
                param_counts[component_name] = 0
            param_counts[component_name] += param_count
        
        # 转换为MB
        total_params_mb = total_params * 2 / (1024 * 1024)  # 假设float16
        
        # 打印参数统计
        for component_name, count in param_counts.items():
            percentage = (count / total_params) * 100
            param_table.add_row([
                component_name,
                f"{count:,} ({count * 2 / (1024 * 1024):.2f} MB)",
                f"{percentage:.2f}%"
            ])
        
        param_table.add_row([
            "总计",
            f"{total_params:,} ({total_params_mb:.2f} MB)",
            "100.00%"
        ])
        
        print(param_table)
        self.total_params = total_params
    
    def print_model_hierarchy(self):
        """打印模型层次结构"""
        print("\n=== 模型层次结构 ===")
        
        def print_module(module, prefix="", is_last=True):
            """递归打印模块结构"""
            # 获取模块名称
            module_name = module.__class__.__name__
            
            # 打印前缀和模块名称
            branch = "└── " if is_last else "├── "
            print(f"{prefix}{branch}{module_name}")
            
            # 准备子模块的前缀
            if is_last:
                child_prefix = f"{prefix}    "
            else:
                child_prefix = f"{prefix}│   "
            
            # 获取子模块
            try:
                children = list(module.named_children())
                if children:
                    for i, (name, child) in enumerate(children):
                        # 跳过权重参数，只关注模块结构
                        if isinstance(child, torch.nn.Module):
                            print_module(child, child_prefix, i == len(children) - 1)
            except AttributeError:
                # 某些对象可能没有named_children方法
                pass
        
        # 从模型开始递归打印
        print_module(self.model)
    
    def print_layer_details(self):
        """打印每层的详细信息"""
        print("\n=== 每层详细信息 ===")
        
        # 尝试从model.model获取层信息（transformers的常见结构）
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            print(f"共有 {len(layers)} 个Decoder层")
            
            # 只打印前3层和后3层的详细信息，避免输出过多
            display_layers = []
            if len(layers) <= 6:
                display_layers = layers
            else:
                display_layers.extend(layers[:3])
                display_layers.extend(layers[-3:])
                
            # 打印层信息
            layer_table = PrettyTable()
            layer_table.field_names = ["层索引", "组件", "参数数量", "输入形状", "输出形状"]
            
            for i, layer in enumerate(display_layers):
                if i >= 3 and i < len(display_layers) - 3:
                    continue
                    
                # 计算层参数数量
                layer_params = sum(p.numel() for p in layer.parameters())
                
                # 获取层组件
                components = []
                for name, component in layer.named_children():
                    components.append(name)
                
                layer_table.add_row([
                    i if i < 3 or i >= len(layers) - 3 else f"...{i}",
                    ", ".join(components),
                    f"{layer_params:,}",
                    "[batch_size, seq_len, hidden_size]",
                    "[batch_size, seq_len, hidden_size]"
                ])
            
            print(layer_table)
        else:
            print("未找到标准的layers结构")
    
    def analyze_kv_cache(self):
        """分析KV缓存相关组件"""
        print("\n=== KV缓存分析 ===")
        
        # 在Qwen3模型中，KV缓存通常在attention层中实现
        attention_layers = []
        
        def find_attention_layers(module, path="model"):
            """递归查找注意力层"""
            try:
                for name, child in module.named_children():
                    current_path = f"{path}.{name}" if path else name
                    
                    # 检查是否为注意力层
                    if "attention" in name.lower() or "attn" in name.lower():
                        attention_layers.append((current_path, child))
                    
                    # 递归查找
                    find_attention_layers(child, current_path)
            except AttributeError:
                pass
        
        # 开始查找
        find_attention_layers(self.model)
        
        if attention_layers:
            print(f"找到 {len(attention_layers)} 个注意力层")
            
            # 打印前几个注意力层的详细信息
            kv_cache_table = PrettyTable()
            kv_cache_table.field_names = ["路径", "类型", "头数", "KV头数", "头维度"]
            
            display_count = min(5, len(attention_layers))
            for i in range(display_count):
                path, layer = attention_layers[i]
                
                # 获取注意力头相关参数
                num_heads = getattr(layer, "num_heads", 
                                  getattr(layer, "num_attention_heads", "N/A"))
                num_kv_heads = getattr(layer, "num_kv_heads", 
                                     getattr(layer, "num_key_value_heads", "N/A"))
                head_dim = getattr(layer, "head_dim", "N/A")
                
                kv_cache_table.add_row([
                    path,
                    layer.__class__.__name__,
                    num_heads,
                    num_kv_heads,
                    head_dim
                ])
            
            print(kv_cache_table)
            
            # 说明KV缓存的工作机制
            print("\nKV缓存工作机制:")
            print("1. Qwen3模型使用多头注意力机制，通常采用分组查询注意力(GQA)")
            print("2. KV缓存用于存储先前计算的键和值，避免重复计算")
            print("3. 在自回归生成过程中，只需要计算当前token的查询(Q)，然后与历史KV缓存进行注意力计算")
            print("4. 缓存大小与序列长度、头数和头维度成正比")
        else:
            print("未找到明显的注意力层")
    
    def save_model_info_to_file(self):
        """保存模型信息到文本文件"""
        output_file = "qwen3_model_structure_info.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Qwen3模型结构信息 ===\n\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"模型类型: {self.model.__class__.__name__}\n")
            f.write(f"设备: {next(self.model.parameters()).device}\n")
            f.write(f"总参数量: {self.total_params:,} ({self.total_params * 2 / (1024 * 1024):.2f} MB)\n\n")
            
            # 保存配置信息
            f.write("=== 模型配置 ===\n")
            config = self.model.config
            for key, value in config.__dict__.items():
                if not key.startswith("_"):
                    f.write(f"{key}: {value}\n")
        
        print(f"\n模型信息已保存到: {output_file}")

    def test_inference(self):
        """简单的推理测试，验证模型是否正常工作"""
        if self.model is None or self.tokenizer is None:
            print("请先加载模型和分词器")
            return
        
        print("\n=== 推理测试 ===")
        try:
            # 简单的文本生成测试
            prompt = "你好，我是"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"提示词: {prompt}")
            print(f"生成结果: {generated_text}")
        except Exception as e:
            print(f"推理测试失败: {e}")

def main():
    """主函数"""
    # 创建分析器实例
    analyzer = Qwen3ModelAnalyzer(model_path="/opt/llm/Qwen3-0.6B/")
    
    # 加载模型
    if analyzer.load_model():
        # 分析模型结构
        analyzer.analyze_model_structure()
        
        # 执行简单的推理测试
        analyzer.test_inference()
    
    print("\n模型分析完成！")

if __name__ == "__main__":
    main()