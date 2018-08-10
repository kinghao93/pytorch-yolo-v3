# A PyTorch implementation of a YOLO v3 Object Detector



## 运行环境
1. Anaconda3/Python 3.6
2. OpenCV3
3. PyTorch 0.4

PyTorch 0.3 not test!



## 检测示例

![Detection Example](https://i.imgur.com/m2jwneng.png)
## Running the detector

### 一幅或多幅图像检测

将工程克隆下来, 你首先需要下载coco数据集下训练的权重文件 [here](https://pjreddie.com/media/files/yolov3.weights),放在工程根目录下

如果需要下载权重文件, 在Linux下,你还可以这样做

```
wget https://pjreddie.com/media/files/yolov3.weights 
python detect.py --images imgs --det det 
```


`--images`  图片路径或图像目录所在, and `--det` 检测结果保存位置
 其他设置项如 批处理大小 (using `--bs` flag) , 目标检测阈值等都可以在其中设置. 

```
python detect.py -h
```

