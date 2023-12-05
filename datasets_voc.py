import torch
import torchvision
import torchvision.transforms.v2 as v2


voc_dataset = torchvision.datasets.VOCDetection(root='D:\\image\\datasets\\VOC2007', 
                                  year='2007', image_set='test', download=False)
image, target = voc_dataset.__getitem__(1)
print(image)