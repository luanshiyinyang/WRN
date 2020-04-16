"""
Author: Zhou Chen
Date: 2020/4/16
Desc: desc
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from wrn import WideResNet
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
batch_size = 32
layers = 28
epochs = 100

torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, scheduler, epoch):

    model.train()
    for step, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        output = model(input)

        loss = criterion(output, target)
        acc = accuracy(output.data, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print("training step: {}, loss: {}, acc: {}".format(step, loss.item() / input.size(0), acc.item()))

    return loss.item() / input.size(0)


def validate(val_loader, model, criterion, epoch):
    model.eval()

    for step, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)
        acc = accuracy(output.data, target)
    if step % 50 == 0:
        print("training step: {}, loss: {}, acc: {}".format(step, loss.data.item() / input.size(0), acc.item()))
    return acc.item(), loss.data.item() / input.size(0)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    # 保存最好的模型
    if is_best:
        torch.save(state, filename)


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:1].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))

    return res[0] / batch_size


def main(augment=False):
    # 数据准备
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # 是否增广
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ])
    transform_valid = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../data/Caltech101/train/', transform=transform_train),
        batch_size=batch_size,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../data/Caltech101/valid', transform=transform_valid),
        batch_size=batch_size,
        shuffle=False)

    print("the number of train images: {}".format(len(train_loader.dataset)))
    print("the number of valid images: {}".format(len(val_loader.dataset)))

    # 构建模型
    model = WideResNet(layers, k=10, dropout_prob=0.5)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # 转移到GPU
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    his = {
        'train_loss': [],
        'valid_loss': []
    }
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch)
        acc, valid_loss = validate(val_loader, model, criterion, epoch)
        his['train_loss'].append(train_loss)
        his['valid_loss'].append(valid_loss)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'epoch_acc': acc,
        }, is_best)
    print('best accuracy: ', best_acc)
    with open('his.pkl', 'wb') as f:
        pickle.dump(his, f)


if __name__ == '__main__':
    main()
