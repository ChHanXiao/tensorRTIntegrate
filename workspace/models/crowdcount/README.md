# DM-Count PyTorch=>ONNX=>TensorRT

## 1.Reference
- **github:** https://github.com/cvlab-stonybrook/DM-Count

## 2.Export ONNX Model

copy [export_onnx.py](export_onnx.py) 

```
modify forward function in models.py like
def forward(self, x):
    x = self.features(x)
    # x = F.upsample_bilinear(x, scale_factor=2)
    x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = self.reg_layer(x)
    mu = self.density_layer(x)
    if torch.onnx.is_in_onnx_export():
    return mu
    B, C, H, W = mu.size()
    mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = mu / (mu_sum + 1e-6)
    return mu, mu_normed

```

## 3.TRT

**INPUT**

[batch_size,3,1280,1280]

**OUTPUT**

[batch_size,1,160,160]