# pytorch-yolo-v3
一步步理解基于pytorch实现yolo-v3过程

## 001 
YOLO V3 核心网络的搭建以及检测测试

## 002
模型权重加载, 图像批处理,预测结果nms,检测结果可视化
```
python detect.py 
```
默认batch_size=1, 输入`imgs/`中图片, 检测结果保存在 `det/`目录下

示例输出: 

![](002/det/det_dog.jpg)
