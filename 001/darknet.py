"""
python 3.6
Pytorch 0.4
"""
import torch
import torch.nn as nn
from util import predict_transform

def parse_cfg(cfgfile):
    """
    输入: 配置文件路径
    返回值: list对象,其中每一个item为一个dict类型
    对应于一个要建立的神经网络模块
    """

    # 加载文件并过滤掉文本中多余内容
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0] # 去掉空行
    lines = [x for x in lines if x[0]!='#'] # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines] # 去掉左右两边的空格

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # 这是一个层(块)的开始
            # 上一个块内容如果还没有保存
            if len(block) != 0:  # 块内已经存了信息, 都是上一个块的信息
                blocks.append(block)
                block = {}  # 新建一个空白块存描述信息
            block["type"] = line[1:-1].rstrip()  # 块名
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)  # 退出循环，将最后一个未加入的block加进去

    # print('\n\n'.join([repr(x) for x in blocks]))
    return blocks

# 配置文件定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}

# cfg = parse_cfg("cfg/yolov3.cfg")
# print(cfg)


class EmptyLayer(nn.Module):
    """
    为shortcut layer / route layer 准备, 具体功能不在此实现
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    '''yolo 检测层'''
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, input_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, input_dim, self.anchors, num_classes, confidence, CUDA)

        return prediction

def create_modules(blocks):
    # 获取网路输入和预处理相关信息
    net_info = blocks[0]

    module_list = nn.ModuleList()
    index = 0 # route layer 会用到
    previous_filters = 3 # 初始值对应于输入数据3通道
    output_filters = []

    for block in blocks:
        container = nn.Sequential()
        if block["type"] == "net":
            continue

        if block["type"] == "convolutional":
            ''' 1. 卷积层 '''
            # 获取激活函数/批归一化/卷积层参数
            activation = block["activation"]
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 开始创建并添加相应层
            # Add the convolutional layer
            # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(previous_filters, filters, kernel_size, stride, pad, bias=bias)
            container.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                container.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            # 给定参数负轴系数0.1
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                container.add_module("leaky_{0}".format(index), activn)

        elif block["type"] == "upsample":
            '''
            2. upsampling layer
            没有使用 Bilinear2dUpsampling
            实际使用的为最近邻插值
            '''
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            container.add_module("upsample_{}".format(index), upsample)

        # route layer -> Empty layer
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(',')

            #Start  of a route
            start = int(block["layers"][0])
            #end, if there exists one.
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            #Positive anotation: 正值
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            container.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        # shortcut corresponds to skip connection
        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            shortcut = EmptyLayer()
            container.add_module("shortcut_{}".format(index), shortcut)

        elif block["type"] == "maxpool":
            stride = int(block["stride"])
            size = int(block["size"])
            maxpool = nn.MaxPool2d(size, stride)
            container.add_module("maxpool_{}".format(index), maxpool)

        # Yolo is the detection layer
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors) # 锚点,检测,位置回归,分类
            container.add_module("Detection_{}".format(index), detection)
        else:
            print("...咱未实现的...")
            assert False

        module_list.append(container)
        previous_filters = filters
        output_filters.append(filters)
        index += 1

    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        # 模型版本标志
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks # list

    def get_module_list(self):
        return self.module_list # nn.ModuleList

    def forward(self, x, CUDA=True):
        detections = []
        # 除了net块之外的所有
        modules = self.blocks[1:]

        # cache output for route layer
        outputs = {}

        write = False # 拼接检测层结果
        for i in range(len(modules)):
            module_type = modules[i]["type"]

            #
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
            #
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i - 1] + outputs[i + from_]  # 求和运算
                outputs[i] = x

            #
            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(modules[i]["classes"])

                # Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if type(x) == int:
                    continue

                # 将在3个不同level的fm上检测结果
                # 存储在 detections 里
                if not write:
                    detections = x
                    write = True

                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]
        # 网络forward 执行完毕
        try:
            return detections
        except:
            return 0

# blocks = parse_cfg('cfg/yolov3.cfg')
# x,y = create_modules(blocks)
# print(y)

model = Darknet("cfg/yolov3.cfg")
input = torch.sigmoid(torch.rand(1, 3, 416, 416).float())
# 网络输入数据大小
model.net_info["height"] = 416
predictions = model(input, False)
print(predictions.shape) # torch.Size([1, 10647, 85]) 10647 = ( 13*13 + 26*26 + 52*52) * 3 [anchors]