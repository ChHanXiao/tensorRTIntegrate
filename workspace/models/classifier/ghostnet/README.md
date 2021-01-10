# ghostnet PyTorch=>ONNX=>TensorRT

## 1.Reference
- **arxiv:** [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
- **github:** [https://github.com/huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet)
- get ghostnet weights from here: [ghostnet/pytorch](https://github.com/huawei-noah/ghostnet/blob/master/pytorch/models/state_dict_93.98.pth)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.TRT

**INPUT**

[batch_size,3,224,224]

**OUTPUT**

[batch_size,1000]

## 4.Results

![](prediction.jpg)