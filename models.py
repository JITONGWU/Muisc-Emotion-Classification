'''
model_archive.py

A file that contains neural network models.
You can also implement your own model here.
'''
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, hparams):
        super(Baseline, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2), #padding = (kernel_size - 1)/2 to keep size same
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(13)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(19)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
   
        self.linear = nn.Linear(64*9, hparams.genres)

    def forward(self, x):
        #print(x.shape)
        x = x.float()
        x = x.view(x.size(0),1, x.size(1))

        #x = x.transpose(1, 2)
   #     print(x.shape)

        x = self.conv0(x)
  #      print(x.shape)
        x = self.conv1(x)
 #       print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = x.view(x.size(0), x.size(1)*x.size(2))
        x = self.linear(x)

        return x
