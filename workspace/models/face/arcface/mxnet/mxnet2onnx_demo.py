import argparse
import onnx
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
from onnx import numpy_helper
from onnx import TensorProto
from onnx import helper

print('mxnet version:', mx.__version__)
print('onnx version:', onnx.__version__)

parser = argparse.ArgumentParser(description='convert arcface models to onnx')
# general
parser.add_argument('--prefix', default='./mobilefacenet-res2-6-10-2-dim512/model', help='prefix to load model.')
parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
parser.add_argument('--input_shape', nargs='+', default=[1, 3, 112, 112], type=int, help='input shape.')
parser.add_argument('--output_onnx', default='./arcface.onnx', help='path to write onnx model.')
args = parser.parse_args()

input_shape = args.input_shape
print('input-shape:', input_shape)

# BN bug fix?
sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
all_layers = sym.get_internals()
for layer in all_layers:
    if 'gamma' in layer.name and layer.attr('fix_gamma') == 'True':
        arg_params[layer.name] = mx.nd.array(np.ones(arg_params[layer.name].shape))
mx.model.save_checkpoint(args.prefix + "r", args.epoch, sym, arg_params, aux_params)

sym_file = f'{args.prefix + "r"}-symbol.json'
params_file = f'{args.prefix + "r"}-{args.epoch :04d}.params'
onnx_r = args.output_onnx.split('.onnx')[0]+'-tmp.onnx'
converted_model_path = onnx_mxnet.export_model(sym_file, params_file, [input_shape], np.float32, onnx_r)

# Check the model
onnx.checker.check_model(onnx_r)
print('The onnx_r is checked!')

def createGraphMemberMap(graph_member_list):
    member_map = dict()
    for n in graph_member_list:
        member_map[n.name] = n
    return member_map

model = onnx.load_model(onnx_r)
graph = model.graph
input_map = createGraphMemberMap(model.graph.input)

# ===PReLU slop===
# C--->C*1*1
for input_name in input_map.keys():
    if input_name.endswith('relu_gamma'):
        # print(input_name)
        input_shape_tmp = input_map[input_name].type.tensor_type.shape.dim
        input_dim_val = input_shape_tmp[0].dim_value
        # print(input_dim_val)

        graph.input.remove(input_map[input_name])
        new_nv = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [input_dim_val, 1, 1])
        graph.input.extend([new_nv])

        initializer_map = createGraphMemberMap(graph.initializer)
        graph.initializer.remove(initializer_map[input_name])
        weight_array = numpy_helper.to_array(initializer_map[input_name])
        # print(weight_array.shape)
        weight_array = weight_array.astype(np.float)  # np.float32与initializer中不匹配
        # ===可以查看权重输出！！！
        b = []
        for w in weight_array:
            b.append(w)
        new_nv = helper.make_tensor(input_name, TensorProto.FLOAT, [input_dim_val, 1, 1], b)
        graph.initializer.extend([new_nv])

onnx_static = args.output_onnx
onnx.save_model(model, onnx_static)
onnx.checker.check_model(onnx_static)
print('The onnx_static is checked!')

if 'data' in input_map:
   data_indx = list(input_map).index('data')

d = model.graph.input[data_indx].type.tensor_type.shape.dim
rate = (input_shape[2] / d[2].dim_value, input_shape[3] / d[3].dim_value)
print("rate: ", rate)
d[0].dim_param = 'batch_size'
d[2].dim_value = int(d[2].dim_value * rate[0])
d[3].dim_value = int(d[3].dim_value * rate[1])
for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    d[0].dim_param = 'batch_size'
onnx_dynamic = args.output_onnx.split('.onnx')[0]+'-d.onnx'
onnx.save_model(model, onnx_dynamic)
onnx.checker.check_model(onnx_dynamic)
print('The onnx_dynamic is checked!')
