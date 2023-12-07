import datasets_voc
import predict_image_box
import torch
import torchvision
import os
import torchvision.transforms.v2 as v2

BATCH_SIZE = 1
EPOCH = 1

if __name__ == '__main__':
    voc_dataset_image_transform = v2.Compose([
        datasets_voc.BoxScale(224, 224),
        v2.ToImageTensor(),
        v2.ConvertImageDtype(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    voc_trainval_dataset = datasets_voc.VOCDetectionD(root='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
                                                      year='2007', image_set='train', download=False,
                                                      transforms=voc_dataset_image_transform)
    voc_train_val_dataset_loader = torch.utils.data.DataLoader(dataset=voc_trainval_dataset, batch_size=2, shuffle=True,
                                                               collate_fn=datasets_voc.VocDatasetCollater())

    for epoch in range(EPOCH):
        for (image, target) in voc_train_val_dataset_loader:

            predict_image_box.faster_rcnn_frame.train()
            loss = predict_image_box.faster_rcnn_frame(image, target)

            loss_classifier = loss['loss_classifier']
            loss_box_reg = loss['loss_box_reg']
            loss_objectness = loss['loss_objectness']
            loss_rpn_box_reg = loss['loss_rpn_box_reg']

            print(loss_classifier, '\t', loss_box_reg, '\t', loss_objectness, '\t', loss_rpn_box_reg)
