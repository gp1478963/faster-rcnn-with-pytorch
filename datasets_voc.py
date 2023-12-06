from typing import Any, Tuple

import numpy
import torch
import torchvision
import torchvision.transforms.v2 as v2
from PIL.Image import Image
import xml.etree.ElementTree as ET
classicers = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class BoxScale:
    def __call__(self, *args, **kwargs):
        pass


voc_dataset_image_transform = v2.Compose([
    v2.ToImageTensor(),
    v2.ConvertImageDtype(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.Resize((224, 224))
])

voc_trainval_dataset = torchvision.datasets.VOCDetection(root='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
                                                         year='2007', image_set='train', download=False)
voc_train_val_dataset_loader = torch.utils.data.DataLoader(voc_trainval_dataset, batch_size=1, shuffle=True)

image, target = voc_trainval_dataset.__getitem__(1)


class VOCDetectionD(torchvision.datasets.VOCDetection):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET.ET_parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


if __name__ == '__main__':
    import predict_image_box
    (image_size_original_h, image_size_original_w) = image.size
    # image size调整到（224， 224）尺寸下需要缩放的倍数，box一样需要调整
    scale_h, scale_w = 224 / image_size_original_h, 224 / image_size_original_w
    image1, target = voc_dataset_image_transform(image, target)
    image1 = torch.unsqueeze(image1, dim=0)

    boxes = []
    labels = []
    for image_target in target['annotation']['object']:
        x1, y1, x2, y2 = (int(image_target['bndbox']['xmin']), int(image_target['bndbox']['ymin']),
                          int(image_target['bndbox']['xmax']), int(image_target['bndbox']['ymax']))
        label = classicers.index(image_target['name'])

        boxes.insert(-1, [x1, y1, x2, y2])
        labels.insert(-1, label)

    boxes = torch.from_numpy(numpy.array(boxes)).float()
    labels = torch.from_numpy(numpy.array(labels, dtype=numpy.int64))

    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_h
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_w
    target = []
    target.insert(-1, {'boxes': boxes, 'labels': labels})
    # 至此，标签制作完成
    predict_image_box.faster_rcnn_frame.train()

    loss_classifier,  loss_box_reg, loss_objectness, loss_rpn_box_reg = predict_image_box.faster_rcnn_frame(image1, target)
    # predictions_boxs, predictions_labels, predictions_scores = predictons['boxes'], predictons['labels'], predictons['scores'],
    # keep_index = torchvision.ops.nms(predictions_boxs, predictions_scores, iou_threshold=0.1)
    # print(predictions_boxs[keep_index])
    # print(predictions_labels[keep_index])













