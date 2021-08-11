import torch
import torch.nn as nn

class SpokenDigitModel(nn.Module):
    
    def __init__(self, input_size=1, output_size=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.flatten_layer = nn.Sequential(
            nn.Flatten()
        )
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(64, output_size),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten_layer(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x

#https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
class FeatureExtractor(nn.Module):
    def __init__(self, cnn_model):
        super().__init__()
        self.cnn_model = cnn_model
        self._features = []
        # self.cnn_model.fc_layer1.register_forward_hook(self._save_outputs_hook())
        # self.cnn_model.flatten_layer.register_forward_hook(self._save_outputs_hook())
        self.cnn_model.fc_layer2.register_forward_hook(self._save_outputs_hook())

    def _save_outputs_hook(self):
        def fn(_, __, output):
            self._features = output.tolist()
        return fn

    def forward(self, x):
        self.cnn_model(x)
        return self._features

if __name__ == '__main__':
    from torchsummary import summary

    model = SpokenDigitModel()
    print(model)
    
    summary(model,(1,55,55),device="cpu")









