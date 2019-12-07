import torch
import torch.nn as nn
import torchvision

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # layer 2
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # layer 3
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # # layer 4
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # # layer 5
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True)
            )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 100)
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 10)
            # nn.Softmax(dim=1)
            )

        self.relu_locs = {}

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            # print(layer)
            # print('\t', x.size(), end='---->')
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.relu_locs[idx] = location
                # print(x.size())
            else:
                x = layer(x)
            # print(x.size())

        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output


class Nnet(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LocalResponseNorm(64),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.dense = nn.Sequential(
            nn.Linear(512, 10))

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        linear_input = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(linear_input)
        return out

if __name__ == '__main__':
    # model = net()
    # print(model(torch.randn(5, 3, 32, 32)))
    # x = torch.randn(3,4,5)
    # print(x[1,:,:].view(1,4,5).size())

    model = torchvision.models.vgg13()
    print(model)