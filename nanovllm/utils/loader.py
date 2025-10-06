import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # packed_modules_mapping 的功能是将 packed_modules 中的参数名映射到 safetensors 中的参数名
    # 因为这里做了一些算子融合，那就和直接加载 safetensors 中的参数名不同了
    # 例如 这里将 gate_proj 和 up_proj 融合到了一个算子 gate_up_proj中
    # 那么加载到satetensors的gate_proj时，就要将名字replace成gate_up_proj,然后让模型的gate_up_proj加载这个权重
    # gate_up_proj调用自己的weight_loeader，传入'model.layers.0.mlp.gate_proj.weight' 以及 shard_id 0 来加载权重
    # 其余普通的参数就直接model.get_parameter(weight_name)来获取参数，然后调用默认的weight_loader来加载权重
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name) # nn.Module内置api，来获取参数 将param_name按最后一个.分割为 module_name.param
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
