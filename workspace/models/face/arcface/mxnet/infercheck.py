import onnxruntime as ort
import cv2
import numpy as np
import torch
import onnx
from onnxsim import simplify

# model = onnx.load('./arcface.onnx')
# model_simp, check = simplify(model)
# assert check, "Simplified ONNX model could not be validated"
# onnx_simplify = './arcface-simplify.onnx'
# onnx.save_model(model, onnx_simplify)
# onnx.checker.check_model(onnx_simplify)

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 112
    height_new = 112
    # 判断图片的长宽比率

    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


image_path = "./test1.jpg"
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_raw = img_resize(img_raw)
img = np.float32(img_raw)
# img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)
ort_session = ort.InferenceSession('./arcface-simplify.onnx')
#
facefeature = ort_session.run(None, {'data': img})
#
print(facefeature)