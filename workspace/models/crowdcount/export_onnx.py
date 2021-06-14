import torch
from models import vgg19

model_path = "pretrained_models/model_qnrf.pth"
device = torch.device('cuda:0')  # device can be "cpu" or "gpu"
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
dummy_input = torch.randn(1, 3, 1280, 1280).to(device)
onnx_output = 'dm_count.onnx'
input_names = ['input']
output_names = ['output']
dynamic_axes = {"input": {0: "batch_size"},
              "output": {0: "batch_size"}}
torch.onnx.export(model, dummy_input, onnx_output, verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=12,
                  dynamic_axes=dynamic_axes)