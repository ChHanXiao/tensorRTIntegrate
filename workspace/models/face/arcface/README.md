# ArcFace Network MxNet=>ONNX=>TensorRT

## 1.Reference
- **insightface github:** https://github.com/deepinsight/insightface
- **mnn_example:** https://github.com/MirrorYuChen/mnn_example

## 2.Export ONNX Model

- copy [mxnet2onnx_demo.py](./mxnet/mxnet2onnx_demo.py) to `insightface/recognition/ArcFace` 

```
python3 mxnet2onnx_demo.py
```

## 3.TRT

**INPUT**

[batch_size,3,112,112]

**OUTPUT**

[batch_size,512]

## 4.Results

- retinaface result

![](./prediction.jpg)

- face location|face_aligned

![](./0.img_face.jpg)![](./0.face_aligned.jpg)

- arcface feature 512

![](./arcfeature.jpg)