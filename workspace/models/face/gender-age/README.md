# gender-age |alignment MxNet=>ONNX=>TensorRT

## 1.Reference
- **insightface github:** https://github.com/deepinsight/insightface

## 2.Export ONNX Model

- copy [mxnet2onnx_demo.py](./mxnet/mxnet2onnx_demo.py) to `insightface/gender-age` 

```
python3 mxnet2onnx_demo.py
```

[fixed export onnx](https://zhuanlan.zhihu.com/p/166267806): BN / PReLU slop

## 3.TRT

gender-age

**INPUT**

[batch_size,3,112,112]

**OUTPUT**

[batch_size,202]

landmark

**INPUT**

[batch_size,3,192,192]

**OUTPUT**

[batch_size,212]

## 4.Results

- gender-age |  landmark 106

![](./0.face_aligned.jpg)![](./1.face_aligned.jpg)