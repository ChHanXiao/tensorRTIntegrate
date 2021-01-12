import onnx
import torch
from ghostnet import ghostnet

model = ghostnet()
model.load_state_dict(torch.load('./models/state_dict_93.98.pth'))
image = torch.zeros(1, 3, 224, 224)
dynamic_axes={"input": {0: "batch_size"},
              "output": {0: "batch_size"}}
torch.onnx.export(model, image, "ghostnet.onnx", opset_version=12,
                  input_names=['input'], output_names=['output'])

onnx_model = onnx.load("ghostnet.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)
