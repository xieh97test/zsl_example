import torch
import torch.nn as nn


def init_layer(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=0.1)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class Baseline(nn.Module):

    def __init__(self, in1_features, in2_features, hidden_units):
        super(Baseline, self).__init__()

        self.fc1 = nn.Linear(in1_features, hidden_units, bias=False)
        self.fc2 = nn.Linear(hidden_units, in2_features, bias=False)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1, x2):
        """
        :param x1: (num_samples, num_segments, in1_features)
        :param x2: (num_classes, in2_features)
        :return: (num_samples, num_classes)
        """

        x1 = torch.mean(x1, dim=1, keepdim=False)  # (num_samples, in1_features)

        x1 = torch.tanh(self.fc1(x1))
        x1 = self.fc2(x1)  # (num_samples, in2_features)

        x2 = x2.transpose(0, 1)  # (in2_features, num_classes)

        output = x1.matmul(x2)  # (num_samples, num_classes)

        return output
