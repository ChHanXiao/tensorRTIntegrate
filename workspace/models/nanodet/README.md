# Nanodet PyTorch=>ONNX=>TensorRT

## 1.Reference
- **nanodet:** [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)
- get nanodet pretrained weights from here: [COCO pretrain weight for torch>=1.6(Google Drive)](https://drive.google.com/file/d/1EhMqGozKfqEfw8y9ftbi1jhYu86XoW62/view?usp=sharing) | [COCO pretrain weight for torch<=1.5(Google Drive)](https://drive.google.com/file/d/10h-0qLMCgYvWQvKULqbkLvmirFR-w9NN/view?usp=sharing)

## 2.Export ONNX Model
```
git clone https://github.com/RangiLyu/nanodet.git
```
copy [export_onnx.py](export_onnx.py) into `nanodet/tools` and run `export_onnx.py` to generate `nanodet-m.onnx`.
```
python3 tools/export_onnx.py
```

## 3.TRT

**INPUT**

[batch_size,3,320,320]

**OUTPUT**

[batch_size,1600,80]

[batch_size,400,80]

[batch_size,100,80]

[batch_size,1600,32]

[batch_size,400,32]

[batch_size,100,32]

## 4.Results

![](prediction.jpg)