import torch
import numpy
import torchvision as vision
import torchvision.transforms.v2 as v2
import cv2


model_resudal_neutral_50_layer = vision.models.resnet50(weights=vision.models.ResNet50_Weights.IMAGENET1K_V1)
backbone_resudal_neutral_50_layer = torch.nn.Sequential(
            model_resudal_neutral_50_layer.conv1,
            model_resudal_neutral_50_layer.bn1,
            model_resudal_neutral_50_layer.relu,
            model_resudal_neutral_50_layer.maxpool,
            model_resudal_neutral_50_layer.layer1,
            model_resudal_neutral_50_layer.layer2,
            model_resudal_neutral_50_layer.layer3,
            model_resudal_neutral_50_layer.layer4,
            model_resudal_neutral_50_layer.avgpool,
)
backbone_resudal_neutral_50_layer.out_channels = 2048

anchor_generator = vision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                  aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = vision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

faster_rcnn_frame = vision.models.detection.FasterRCNN(
    backbone=backbone_resudal_neutral_50_layer, 
    num_classes=20,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

faster_rcnn_frame.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

predictions = faster_rcnn_frame(x)
print(predictions)


