import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np

import sys
import os
import pdb

from models import net


np.random.seed(0)
torch.manual_seed(1)
torch.cuda.set_device(1)

BATCH_SIZE = 64
LEARNING_RATE = 1e-2
EPOCHS = 20

def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)

def compute_accuracy(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    equal = np.argmax(pred, 1) == label
    return np.mean(np.float32(equal))

def train(net, cuda=False):
    # file = open('train_log.txt', 'w')
    # sys.stdout = file
    if cuda:
        net = net.cuda()
    # net.train()

    # load model if exist
    # if os.path.isfile('model.pth'):
    #     net.load_state_dict(torch.load('model.pth'))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = torchvision.datasets.CIFAR10(root='Cifar-10', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]))
    test_data = torchvision.datasets.CIFAR10(root='Cifar-10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [11, 16], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # print(type(labels))
            # print(inputs.size())
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if cuda:
                loss = loss.cuda()
            loss.backward()
            optimizer.step()
            
            training_loss = loss.item()

            if i % 100 == 0:
                accuracy = compute_accuracy(outputs, labels)
                print("Epoch[{}/{}] iter {:>3d}  loss:{:.4f}  accuracy:{:.4f}".format(epoch, EPOCHS, i, training_loss, accuracy))
        
        lr = scheduler.get_lr()[0]
        scheduler.step()

        net.eval()
        accuracys = []
        for i, data in enumerate(test_loader):
            inputs, labels = data
            # print(inputs.size())
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            accuracys.append(compute_accuracy(outputs, labels))
        print('lr={},---loss on the test dataset: {:.4f}------ '.format(lr, np.mean(accuracys)))
        # print('------loss on the test dataset: {:.4f}------ '.format(np.mean(accuracys)))
            
    torch.save(net.state_dict(), 'model.pth')
    # file.close()


if __name__ == '__main__':
    net = net()
    net.apply(weight_init)
    # print(net.features)
    # pdb.set_trace()
    train(net, torch.cuda.is_available())
