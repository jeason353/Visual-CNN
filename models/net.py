import torch
import torch.nn as nn
import torchvision

class net(nn.Module):
    def __init__(self, pretrained=True):
        super(net, self).__init__()

        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),            
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # layer 2
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # layer 3
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # # layer 4
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # # layer 5
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True)
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(4096, 1000)
            )

        self.feature_maps = {}
        self.pool_indices = {}
            
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_indices[idx] = location
            else:
                x = layer(x)
            self.feature_maps[idx] = x
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output


if __name__ == '__main__':
    model = net()
    print(model)
