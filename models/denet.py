import torch.nn as nn

class denet(nn.Module):
    def __init__(self, net):
        super(denet, self).__init__()

        self.features = nn.Sequential(
            # layer 1
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),

            # layer 2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),

            # layer 3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),

            # layer 4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 2, stride=1, padding=1),

            # layer 5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.conv2deconv = {}
        
        for i, layer in enumerate(net.features):
            if isinstance(layer, nn.Conv2d):
                self.conv2deconv[i] = len(net.features) - 1 - i

        self.init_weight(net)
    
    def init_weight(self, net):
        for i, layer in enumerate(net.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv[i]].weight.data = layer.weight.data

    def forward(self,):
        