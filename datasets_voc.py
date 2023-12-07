from typing import Any, Tuple

import numpy
import torch
import torchvision
import torchvision.transforms.v2 as v2
import xml.etree.ElementTree as ET
from PIL import Image

classicers = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class BoxScale:
    def __init__(self, adjust_post_h=448, adjust_post_w=448):
        self.adjust_post_h = adjust_post_h
        self.adjust_post_w = adjust_post_w

    def __call__(self, *args, **kwargs):
        image = args[0][0]
        boxes = args[0][1]
        labels = args[0][2]

        image_orig_h, image_orig_w = image.size
        scale_h = self.adjust_post_h / image_orig_h
        scale_w = self.adjust_post_w / image_orig_w

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h

        image = v2.Resize((224, 224))(image)

        return (image, boxes, labels) + args[3:]


def _boxes_and_labels_adjust(target):
    boxes_ = []
    labels_ = []

    for image_target_t in target['annotation']['object']:
        x1_, y1_, x2_, y2_ = (int(image_target_t['bndbox']['xmin']), int(image_target_t['bndbox']['ymin']),
                              int(image_target_t['bndbox']['xmax']), int(image_target_t['bndbox']['ymax']))
        label = classicers.index(image_target_t['name'])

        boxes_.insert(-1, [x1_, y1_, x2_, y2_])
        labels_.insert(-1, label)

    boxes_ = torch.from_numpy(numpy.array(boxes_)).float()
    labels_ = torch.from_numpy(numpy.array(labels_, dtype=numpy.int64))
    return boxes_, labels_


class VOCDetectionD(torchvision.datasets.VOCDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target_ = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        boxes_, labels_ = _boxes_and_labels_adjust(target_)
        if self.transforms is not None:
            img, boxes_, labels_ = self.transforms(img, boxes_, labels_)

        target_ = {'boxes': boxes_, 'labels': labels_}
        return img, target_


class VocDatasetCollater:
    def __init__(self): pass

    def __call__(self, data):
        image_list = []
        target_list = []
        for sampler in data:
            image_list.insert(-1, sampler[0])
            target_list.insert(-1, sampler[1])
        return image_list, target_list


if __name__ == '__main__':
    import predict_image_box

    voc_dataset_image_transform = v2.Compose([
        BoxScale(224, 224),
        v2.ToImageTensor(),
        v2.ConvertImageDtype(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    voc_trainval_dataset = VOCDetectionD(root='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
                                         year='2007', image_set='train', download=False,
                                         transforms=voc_dataset_image_transform)
    voc_train_val_dataset_loader = torch.utils.data.DataLoader(voc_trainval_dataset, batch_size=2, shuffle=True,
                                                               collate_fn=VocDatasetCollater())

    for batch_sampler in voc_train_val_dataset_loader:
        break

    # print(loss_classifier, '\t', loss_box_reg, '\t', loss_objectness, '\t', loss_rpn_box_reg)
    # predictions_boxs, predictions_labels, predictions_scores = predictons['boxes'], predictons['labels'], predictons['scores'],
    # keep_index = torchvision.ops.nms(predictions_boxs, predictions_scores, iou_threshold=0.1)
    # print(predictions_boxs[keep_index])
    # # print(predictions_labels[keep_index])
