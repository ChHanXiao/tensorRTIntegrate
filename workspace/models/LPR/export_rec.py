import torch
from LP_rec import my_lprnet, CHARS

model_path = 'my_lprnet_model.pth'
device = torch.device("cpu")
lprnet = my_lprnet(len(CHARS))
lprnet.to(device)
lprnet.load_state_dict(torch.load(model_path, map_location=device))
dummy_input = torch.randn(1, 3, 24, 94).to(device)
onnx_output = 'lprnet.onnx'
input_names = ['input']
output_names = ['output']
dynamic_axes = {"input": {0: "batch_size"},
              "output": {0: "batch_size"}}
torch.onnx.export(lprnet, dummy_input, onnx_output, verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=12,
                  dynamic_axes=dynamic_axes)
