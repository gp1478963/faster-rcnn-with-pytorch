import datasets_voc
import predict_image_box
import torch
import torchvision
import os
import torch_directml
import torchvision.transforms.v2 as v2

BATCH_SIZE = 1
EPOCH = 10


def train(model, train_loader, val_loader, optimizer, criterion, lr_scheduler, device, epoch_count, epoch_total):
    model.to(device)
    losses = 0.0
    model.train()
    for percent, (image, target) in enumerate(train_loader):
        loss = model(image, target)
        loss_classifier = loss['loss_classifier']
        loss_box_reg = loss['loss_box_reg']
        loss_objectness = loss['loss_objectness']
        loss_rpn_box_reg = loss['loss_rpn_box_reg']
        loss_total = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        lr_scheduler.step()
        losses += loss_total
        average_loss = losses / percent
        print('epoch:<{0}/{1}>, batch:<{2}/{3}>, loss:[{4:.4f}], average loss:[{5:.4f}]'.format(
            epoch_count, epoch_total, percent, train_loader.dataset.__len__(), loss_total.item(), average_loss))

    torch.save(model.state_dict(), './out/model_{0}.pth'.format(epoch_count))


if __name__ == '__main__':
    if torch_directml.is_available():
        device = torch_directml.device(0)
    else:
        device = 'cpu'
    device = 'cpu'
    voc_dataset_image_transform = v2.Compose([
        datasets_voc.BoxScale(224, 224),
        v2.ToImageTensor(),
        v2.ConvertImageDtype(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    voc_trainval_dataset = datasets_voc.VOCDetectionD(root='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
                                                      year='2007', image_set='train', download=False,
                                                      transforms=voc_dataset_image_transform, device=device)
    voc_train_val_dataset_loader = torch.utils.data.DataLoader(dataset=voc_trainval_dataset, batch_size=BATCH_SIZE,
                                                               shuffle=True,
                                                               collate_fn=datasets_voc.VocDatasetCollater(device=device))

    model = predict_image_box.generate_faster_rcnn_model(freeze=True)

    params = [param for param in model.parameters() if param.requires_grad]
    sgd_optimizer = torch.optim.SGD(params=params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=sgd_optimizer, step_size=3, )

    for epoch in range(EPOCH):
        train(model=model, train_loader=voc_train_val_dataset_loader,
              val_loader=None, optimizer=sgd_optimizer, criterion=None,
              lr_scheduler=lr_scheduler, device=device, epoch_count=epoch+1, epoch_total=EPOCH)
