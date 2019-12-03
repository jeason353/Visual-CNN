import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.features = nn.Sequential(
			# layer 1
			nn.Conv2d(3, 96, 7, stride=2, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1, return_indices=True),

			# layer 2
			nn.Conv2d(96, 256, 5, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1, return_indices=True),
			
			# layer 3
			nn.Conv2d(256, 384, 3, stride=1, padding=1),
			nn.ReLU(),

			# layer 4
			nn.Conv2d(384, 384, 3, stride=1, padding=1),
			nn.ReLU(),

			# layer 5
			nn.Conv2d(384, 256, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, return_indices=True)
			)

		self.classifier = nn.Sequential(
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, 100),
			nn.Softmax(dim=1)
			)

		self.relu_locs = {}

	def forward(self, x):
		for idx, layer in enumerate(self.features):
			if isinstance(layer, nn.MaxPool2d):
				x, location = layer(x)
				self.relu_locs[idx] = location
				print(x.size())
			else:
				x = layer(x)

		x = x.view(x.size()[0], -1)
		output = self.classifier(x)
		return output

if __name__ == '__main__':
	model = Net()
	x = torch.randn(1, 3, 224, 224)
	output = model(x)