import torch
from transformers import Qwen3Config
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接使用模拟类，避免分布式初始化
print("使用本地模拟类展示Qwen3模型结构...")

# 创建模拟类用于展示结构
class MockAttention:
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])  # 模拟KV缓存
        self.v_cache = torch.tensor([])

class MockQwen3Attention:
    def __init__(self, hidden_size, num_heads, num_kv_heads, **kwargs):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.attn = MockAttention(num_heads, hidden_size//num_heads, (hidden_size//num_heads)**-0.5, num_kv_heads)

class MockQwen3DecoderLayer:
    def __init__(self, config):
        self.config = config
        self.self_attn = MockQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads
        )
        # 添加更多层的详细信息
        class MockLayerNorm:
            def __init__(self, dim, eps=1e-6):
                self.dim = dim
                self.eps = eps
        
        self.input_layernorm = MockLayerNorm(config.hidden_size)
        self.post_attention_layernorm = MockLayerNorm(config.hidden_size)
        
        class MockMLP:
            def __init__(self, hidden_size, intermediate_size):
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
        
        self.mlp = MockMLP(config.hidden_size, config.intermediate_size)

class Qwen3Model:
    def __init__(self, config):
        self.config = config
        # 使用普通列表而不是ModuleList，避免继承问题
        self.layers = [MockQwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        
        # 添加嵌入层和输出层
        class MockEmbedding:
            def __init__(self, vocab_size, hidden_size):
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
        
        self.embed_tokens = MockEmbedding(config.vocab_size, config.hidden_size)
        
        class MockRMSNorm:
            def __init__(self, dim, eps=1e-6):
                self.dim = dim
                self.eps = eps
        
        self.norm = MockRMSNorm(config.hidden_size)
    
    # 添加named_children方法，模拟PyTorch模块的行为
    def named_children(self):
        yield ('embed_tokens', self.embed_tokens)
        yield ('layers', self.layers)
        yield ('norm', self.norm)


def print_model_structure(model, prefix=''):
    """打印模型结构，标记使用KV缓存的层，优化输出以减少重复"""
    # 尝试获取子模块，如果是列表则特殊处理
    try:
        module_list = list(model.named_children())
    except:
        # 如果没有named_children方法，尝试直接访问属性
        module_list = []
        if hasattr(model, '__dict__'):
            for name, attr in model.__dict__.items():
                if not name.startswith('_') and name not in ['config']:
                    module_list.append((name, attr))
    
    # 统计相同类型的组件
    component_counts = {}
    
    # 第一遍：统计组件类型
    for name, module in module_list:
        if name == 'layers' and isinstance(module, list) and len(module) > 0:
            # 特殊处理layers列表
            layer_type = module[0].__class__.__name__
            component_counts[layer_type] = len(module)
        else:
            module_type = module.__class__.__name__
            component_counts[module_type] = component_counts.get(module_type, 0) + 1
    
    # 第二遍：打印组件信息
    printed_components = set()
    
    for name, module in module_list:
        module_type = module.__class__.__name__
        
        # 检测是否为layers列表
        if name == 'layers' and isinstance(module, list):
            # 打印layers列表信息
            if len(module) > 0:
                layer_type = module[0].__class__.__name__
                print(f"{prefix}+-- {name}: [{len(module)}个{layer_type}]")
                
                # 只显示第一层的详细结构作为示例
                if len(module) > 0 and layer_type not in printed_components:
                    print(f"{prefix}    |-- 示例结构:")
                    print_layer_details("0", module[0], prefix + "    ")
                    printed_components.add(layer_type)
                    
                # 如果层数超过1，显示计数信息
                if len(module) > 1:
                    print(f"{prefix}    |-- 其余{len(module)-1}层结构相同")
        else:
            # 打印非列表组件
            if component_counts[module_type] > 1 and module_type in printed_components:
                # 已打印过该类型的组件，不再重复打印详情
                continue
            else:
                # 首次打印该类型的组件
                if component_counts[module_type] > 1:
                    print(f"{prefix}+-- {name}: {module_type} [共{component_counts[module_type]}个]")
                else:
                    print(f"{prefix}+-- {name}: {module_type}")
                
                # 打印组件的详细信息
                if hasattr(module, 'hidden_size'):
                    print(f"{prefix}    |-- 参数: hidden_size={module.hidden_size}")
                elif hasattr(module, 'dim'):
                    print(f"{prefix}    |-- 参数: dim={module.dim}")
                
                # 标记为已打印
                printed_components.add(module_type)


def print_layer_details(name, module, prefix):
    """打印单个层的详细信息，显示所有参数"""
    # 检查该模块是否有k_cache和v_cache属性
    has_kv_cache = hasattr(module, 'k_cache') and hasattr(module, 'v_cache')
    cache_info = " 📌 包含KV缓存" if has_kv_cache else ""
    
    # 打印模块名称和类型
    print(f"{prefix}+-- {name}: {module.__class__.__name__}{cache_info}")
    
    # 打印模块的所有参数信息
    module_params = []
    # 收集所有可能的参数
    if hasattr(module, 'config'):
        if hasattr(module.config, 'num_attention_heads'):
            module_params.append(f"num_heads={module.config.num_attention_heads}")
        if hasattr(module.config, 'num_key_value_heads'):
            module_params.append(f"num_kv_heads={module.config.num_key_value_heads}")
        if hasattr(module.config, 'hidden_size'):
            module_params.append(f"hidden_size={module.config.hidden_size}")
        if hasattr(module.config, 'intermediate_size'):
            module_params.append(f"intermediate_size={module.config.intermediate_size}")
        if hasattr(module.config, 'rms_norm_eps'):
            module_params.append(f"rms_norm_eps={module.config.rms_norm_eps}")
        if hasattr(module.config, 'vocab_size'):
            module_params.append(f"vocab_size={module.config.vocab_size}")
    else:
        if hasattr(module, 'num_heads'):
            module_params.append(f"num_heads={module.num_heads}")
        if hasattr(module, 'head_dim'):
            module_params.append(f"head_dim={module.head_dim}")
        if hasattr(module, 'num_kv_heads'):
            module_params.append(f"num_kv_heads={module.num_kv_heads}")
        if hasattr(module, 'scale'):
            module_params.append(f"scale={module.scale:.4f}")
        if hasattr(module, 'hidden_size'):
            module_params.append(f"hidden_size={module.hidden_size}")
        if hasattr(module, 'dim'):
            module_params.append(f"dim={module.dim}")
        if hasattr(module, 'vocab_size'):
            module_params.append(f"vocab_size={module.vocab_size}")
    
    if module_params:
        print(f"{prefix}    |-- 全部参数: {', '.join(module_params)}")
    
    # 特殊处理MockQwen3DecoderLayer类型的模块
    if isinstance(module, MockQwen3DecoderLayer):
        # 打印DecoderLayer的子模块结构
        print(f"{prefix}    |-- 子模块结构:")
        print(f"{prefix}        |-- self_attn: MockQwen3Attention 📌 包含KV缓存")
        print(f"{prefix}        |-- mlp: MockMLP")
        print(f"{prefix}        |-- input_layernorm: MockLayerNorm")
        print(f"{prefix}        |-- post_attention_layernorm: MockLayerNorm")


def analyze_kv_cache_usage(model):
    """分析模型中KV缓存的使用情况"""
    print("\n=== KV缓存使用分析 ===")
    layers_with_kv_cache = []
    
    def find_kv_cache_layers(module, path=''):
        # 首先尝试使用named_children方法
        try:
            for name, child in module.named_children():
                current_path = f"{path}.{name}" if path else name
                if hasattr(child, 'k_cache') and hasattr(child, 'v_cache'):
                    layers_with_kv_cache.append((current_path, child))
                find_kv_cache_layers(child, current_path)
        except (AttributeError, TypeError):
            # 如果没有named_children方法或调用失败，尝试直接检查属性
            if hasattr(module, '__dict__'):
                for attr_name, attr_value in module.__dict__.items():
                    if not attr_name.startswith('_') and attr_name not in ['config']:
                        current_path = f"{path}.{attr_name}" if path else attr_name
                        if hasattr(attr_value, 'k_cache') and hasattr(attr_value, 'v_cache'):
                            layers_with_kv_cache.append((current_path, attr_value))
                        # 递归检查子属性
                        try:
                            find_kv_cache_layers(attr_value, current_path)
                        except:
                            pass
    
    find_kv_cache_layers(model)
    
    if layers_with_kv_cache:
        print(f"发现 {len(layers_with_kv_cache)} 个使用KV缓存的层：")
        # 只显示前几个和后几个层的详细信息，避免过多重复
        display_count = 3
        if len(layers_with_kv_cache) <= 2 * display_count:
            # 层数量较少时全部显示
            for path, layer in layers_with_kv_cache:
                print_kv_cache_details(path, layer)
        else:
            # 层数量较多时只显示前几个和后几个
            for path, layer in layers_with_kv_cache[:display_count]:
                print_kv_cache_details(path, layer)
            print(f"... 中间 {len(layers_with_kv_cache) - 2*display_count} 层省略 ...")
            for path, layer in layers_with_kv_cache[-display_count:]:
                print_kv_cache_details(path, layer)
    else:
        print("未找到使用KV缓存的层")


def print_kv_cache_details(path, layer):
    """打印KV缓存的详细信息"""
    print(f"- {path}: {layer.__class__.__name__}")
    print(f"  - k_cache形状: {layer.k_cache.shape if layer.k_cache.numel() > 0 else '未初始化'}")
    print(f"  - v_cache形状: {layer.v_cache.shape if layer.v_cache.numel() > 0 else '未初始化'}")
    
    # 打印额外的缓存相关参数
    cache_params = []
    if hasattr(layer, 'num_heads'):
        cache_params.append(f"num_heads={layer.num_heads}")
    if hasattr(layer, 'num_kv_heads'):
        cache_params.append(f"num_kv_heads={layer.num_kv_heads}")
    if hasattr(layer, 'head_dim'):
        cache_params.append(f"head_dim={layer.head_dim}")
    
    if cache_params:
        print(f"  - 缓存参数: {', '.join(cache_params)}")


def simulate_kv_cache_allocation(model, num_kvcache_blocks, block_size, num_kv_heads, head_dim):
    """模拟KV缓存分配过程，适用于模拟类对象"""
    print("\n=== KV缓存分配模拟 ===")
    print("分配参数:")
    print(f"- KV缓存块数量: {num_kvcache_blocks}")
    print(f"- 块大小: {block_size}")
    print(f"- KV头数: {num_kv_heads}")
    print(f"- 头维度: {head_dim}")
    
    # 计算每层缓存大小
    layer_cache_size_mb = (2 * block_size * num_kv_heads * head_dim * 4) / (1024 * 1024)  # 假设float32
    print(f"- 每层缓存大小: 约 {layer_cache_size_mb:.2f} MB")
    
    # 模拟分配过程
    total_allocated_blocks = 0
    layers_with_allocation = 0
    
    def allocate_kv_cache_to_layers(module):
        nonlocal total_allocated_blocks, layers_with_allocation
        
        # 检查当前模块是否为DecoderLayer
        if isinstance(module, MockQwen3DecoderLayer) and hasattr(module, 'self_attn'):
            # 为该层分配KV缓存
            layers_with_allocation += 1
            total_allocated_blocks += 2  # 每个层分配2个块（K和V）
    
    # 开始分配
    # 由于我们的模型是简化的，直接遍历layers列表
    if hasattr(model, 'layers') and isinstance(model.layers, list):
        print(f"  正在为 {len(model.layers)} 个DecoderLayer层分配KV缓存...")
        for i, layer in enumerate(model.layers):
            # 设置层索引以便于跟踪
            layer.layer_index = i
            allocate_kv_cache_to_layers(layer)
        print(f"  所有层KV缓存分配完成")
    
    # 打印分配结果
    print(f"\n分配结果:")
    print(f"- 分配KV缓存的层数: {layers_with_allocation}")
    print(f"- 总共分配的块数: {total_allocated_blocks}")
    print(f"- 总缓存大小: 约 {total_allocated_blocks * block_size * num_kv_heads * head_dim * 4 / (1024 * 1024):.2f} MB")
    print(f"- 剩余可用块数: {num_kvcache_blocks - total_allocated_blocks}")
    print(f"\nKV缓存分布位置:")
    print("- 所有Qwen3DecoderLayer层的self_attn组件都包含独立的KV缓存")
    print("- 每层缓存包含K和V两个部分，分别用于存储键和值的历史信息")
    print("- 缓存采用块结构管理，支持动态扩展和释放")


def main():
    """主函数：加载模型并展示结构"""
    # 创建一个模拟的配置对象
    class MockConfig:
        def __init__(self):
            self.model = "/opt/llm/Qwen3-0.6B/"
            self.tensor_parallel_size = 1
            self.hf_config = Qwen3Config(
                vocab_size=151936,
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=14336,
                hidden_act="silu",
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                tie_word_embeddings=False,
                torch_dtype=torch.float16,
                rope_theta=1000000,
                head_dim=128
            )
            self.gpu_memory_utilization = 0.9
            self.eos = 151643  # Qwen3的eos token id
    
    # 创建配置对象
    config = MockConfig()
    
    print("=== Qwen3模型结构与KV缓存使用分析 ===")
    print(f"模型名称: {config.model}")
    print(f"张量并行大小: {config.tensor_parallel_size}")
    print(f"隐藏层数量: {config.hf_config.num_hidden_layers}")
    print(f"注意力头数: {config.hf_config.num_attention_heads}")
    print(f"KV头数: {config.hf_config.num_key_value_heads}")
    print(f"隐藏层大小: {config.hf_config.hidden_size}")
    
    # 创建Qwen3模型（简化版本）
    print("\n=== 模型结构 ===")
    hf_config = config.hf_config
    
    # 创建一个Qwen3Model实例
    model = Qwen3Model(hf_config)
    
    # 打印模型结构
    print_model_structure(model)
    
    # 分析KV缓存使用情况
    analyze_kv_cache_usage(model)
    
    # 模拟KV缓存分配
    # 假设的参数值，仅用于演示
    num_kvcache_blocks = 100
    block_size = 16
    num_kv_heads = hf_config.num_key_value_heads // config.tensor_parallel_size
    head_dim = hf_config.head_dim
    
    simulate_kv_cache_allocation(model, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
    
    # 再次分析KV缓存使用情况，确认分配
    analyze_kv_cache_usage(model)
    
    print("\n=== KV缓存绑定机制说明 ===")
    print("1. KV缓存在model_runner.py的allocate_kv_cache方法中统一分配")
    print("2. 分配的缓存形状为：[2, 层数, 块数, 块大小, 头数, 头维度]")
    print("3. 其中第一个维度的0表示key缓存，1表示value缓存")
    print("4. 通过遍历模型的所有子模块，将每一层的k_cache和v_cache绑定到对应的缓存位置")
    print("5. 在实际推理过程中，新计算的k和v会通过store_kvcache函数存储到全局缓存中")
    print("6. 这种设计使得多个序列可以共享KV缓存，提高显存利用率和推理效率")


if __name__ == "__main__":
    main()