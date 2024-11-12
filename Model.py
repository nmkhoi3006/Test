import torch.nn as nn
import torch


class CNNModel(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.block_conv1 = self._make_block(in_channels=3, out_channels=8, kernel_size=3)


        self.flatten = nn.Flatten()
        self.fc1 = self.make_fully_connected(in_features=8*16*16, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_class)

    def forward(self, x):
        x = self.block_conv1(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _make_block(self, in_channels, out_channels, kernel_size, padding="same"):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def make_fully_connected(self, in_features, out_features, p=0.5):
        return nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU()
        )


if __name__ == '__main__':
    fake_data = torch.rand(32, 3, 32, 32)
    model = CNNModel()

    prediction = model(fake_data)
    print(prediction.shape)