# TensorRT

---
## 更新3
* 1、支持OnnX的插件开发，并且实现[CenterNet](https://github.com/xingyizhou/CenterNet)的DCNv2插件demo和Inference实现，附有案例
* 2、基于tensorRT7.0的时候，int8已经失效不可用，此次更新主要提出onnx和onnx插件开发新方法，比基于caffemodel更加简洁方便
* 3、不建议使用pytorch->caffemodel->tensorRT，改用pytorch->onnx->tensorRT，对于任何特定需求（例如dcn、例如双线性插值），可以用插件实现
* 4、如果不用这里提供的框架，自己实现onnx插件，这里有[一份指导](README.onnx.plugin.md)，说明了关键点，可以做参考


## 复现centerNetDCN的检测结果
![image1](/workspace/www.dla.draw.jpg)


## 快速使用
```bash
bash getDLADCN.sh
make run
```

---
## 更新 2
* 1、添加[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)和[CenterNet](https://github.com/xingyizhou/CenterNet)的Inference实现，里面包括了（ChannelMultiplication、Clip、DCN、PiexShuffle）几个插件的实现案例
* 2、屏蔽一个代码，在tensorRT7.0时编译报错，final class
* 3、完善ccutil的函数
* 4、centerNet和alphaPose的模型，请[点击下载](http://zifuture.com:1000/fs/25.shared/tensorRT_demo_model_centerNet_and_openPose.zip)

#### Alpha Pose和CenterNet，分别检测框和点
![image0](/workspace/person_draw.jpg)

## 更新 1
* 1、增加对FP16插件的支持，修改Tensor类支持FP16
* 2、修改TestPlugin插件支持FP16和Float两种格式的案例
* 3、把模型带在代码里面了，开箱用   
---


## 环境
* tensorRT7.0.0.11
* opencv3.4.6（可以任意修改为其他版本）
* cudnn7.6.3（可以任意修改为其他版本）
* cuda10.0（可以任意修改为其他版本）
* Visual Studio 2017（可以用其他版本打开，但需要修改对应opencv版本）
* <font color=red>如果要修改版本，你需要下载cuda/cudnn/tensorRT三者同时匹配的版本，因为他们互相存在依赖，否则只要你是cuda10.0就可以很轻易编译这个项目</font>
* 提交的代码中包含了模型
---


## 案例-Inference
```
auto engine = TRTInfer::loadEngine("models/efficientnet-b0.fp32.trtmodel");
float mean[3] = {0.485, 0.456, 0.406};
float std[3] = {0.229, 0.224, 0.225};
Mat image = imread("img.jpg");
engine->input()->setNormMat(0, image, mean, std);
engine->forward();
engine->output(0)->print();
```

---


## 支持
* Linux
* Windows
* 该封装代码是垮平台的

## 说明
* main.cpp里面有3个案例，分别是Int8Caffe、Onnx、Plugin
* 所有lib依赖项，均采用[import_lib.cpp](src/import_lib.cpp)导入，而不是通过配置
* infer文件夹为对TensorRT封装的Inference代码，目的是简化TensorRT前向过程，封装为Engine类和提供友好高性能的Tensor类支持
* plugin文件夹为对plugin的封装，pluginFactory的实现，以及友好的接口，写新插件只需要
  * 1.plugins里面添加插件类，继承自Plugin::TRTPlugin
  * 2.outputDims和enqueue方法，参照[TestPlugin.cu](src/plugin/plugins/TestPlugin.cu)和[TestPlugin.hpp](src/plugin/plugins/TestPlugin.hpp)，指明该插件的返回维度信息，以及插件前向时的运算具体实现，并在cpp/cu底下加上RegisterPlugin(TestPlugin);，参考[TestPlugin.cu](src/plugin/plugins/TestPlugin.cu)，完成注册
* builder文件夹则是对模型转换做封装，int8Caffe模型编译，onnx模型编译，fp32/fp16模型编译，通过简单的接口实现模型编译到trtmodel
