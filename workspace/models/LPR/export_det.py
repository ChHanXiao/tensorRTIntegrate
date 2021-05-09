import torch
import onnx
from onnxsim import simplify
from net import Retina
from utils import cfg_mnet, py_cpu_nms, decode, decode_landm, PriorBox
import numpy as np
import cv2
import os

model_path = 'mnet_plate.pth'
device = torch.device("cpu")
net = Retina(cfg=cfg_mnet)
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()
dummy_input = torch.randn(1, 3, 640, 640).to(device)
onnx_output = 'test.onnx'
input_names = ['input']
output_names = ['output1', 'output2', 'output3']
dynamic_axes = {"input": {0: "batch_size"},
                "output1": {0: "batch_size"},
                "output2": {0: "batch_size"},
                "output3": {0: "batch_size"}}
torch.onnx.export(net, dummy_input, onnx_output, verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=12,
                  dynamic_axes=dynamic_axes)
if False:
    model = onnx.load(onnx_output)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    output_path = 'simp.onnx'
    onnx.save(model_simp, output_path)
    print('finished exporting onnx ')

img_path = 'export/028125-87_110-204&496_524&585-506&564_204&585_210&514_524&496-0_0_5_24_29_33_24_24-52-45.jpg'
priorbox = PriorBox(cfg_mnet)
points_ref = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])
confidence_threshold=0.02
top_k=1000
nms_threshold=0.4
keep_top_k=500
vis_thres=0.6

srcimg = cv2.imread(img_path)
img = srcimg.astype('float32')
im_height, im_width, _ = img.shape
img -= (104, 117, 123)
with torch.no_grad():
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    loc, conf, landms = net(img)  # forward pass
    prior_data = priorbox((im_height, im_width)).to(device)

    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    landms = landms * scale.repeat(2)
    landms = landms.cpu().numpy()

inds = np.where(scores > confidence_threshold)[0]
boxes = boxes[inds]
landms = landms[inds]
scores = scores[inds]

# keep top-K before NMS
order = scores.argsort()[::-1][:top_k]
boxes = boxes[order]
landms = landms[order]
scores = scores[order]

# do NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
keep = py_cpu_nms(dets, nms_threshold)
# keep = nms(dets, self.nms_threshold,force_cpu=self.cpu)
dets = dets[keep, :]
landms = landms[keep]

# keep top-K faster NMS
dets = dets[:keep_top_k, :]
landms = landms[:keep_top_k, :]
dets = np.concatenate((dets, landms), axis=1)
# show image
for b in dets:
    if b[4] < vis_thres:
        continue
    b = list(map(int, b))
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    img_box = srcimg[y1:y2 + 1, x1:x2 + 1, :]
    new_x1, new_y1 = b[9] - x1, b[10] - y1
    new_x2, new_y2 = b[11] - x1, b[12] - y1
    new_x3, new_y3 = b[7] - x1, b[8] - y1
    new_x4, new_y4 = b[5] - x1, b[6] - y1

    # 定义对应的点
    points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
    # 计算得到转换矩阵
    M = cv2.getPerspectiveTransform(points1, points_ref)
    # 实现透视变换转换
    processed = cv2.warpPerspective(img_box, M, (94, 24))
    cv2.rectangle(srcimg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    # landms
    cv2.circle(srcimg, (b[5], b[6]), 2, (255, 0, 0), thickness=5)
    cv2.circle(srcimg, (b[7], b[8]), 2, (255, 0, 0), thickness=5)
    cv2.circle(srcimg, (b[9], b[10]), 2, (255, 0, 0), thickness=5)
    cv2.circle(srcimg, (b[11], b[12]), 2, (255, 0, 0), thickness=5)
    cv2.imwrite('export/det_LP.jpg', srcimg)


