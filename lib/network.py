import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
########                        TASK 1                            ########
##########################################################################
# Implement Fully-connected neural network with two layers               #
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_out):
        """
        Arguments:
            n_input: dimensionality of input space
            n_hidden: width of hidden layer
            n_out: dimensionality of output space
        """
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        return x
##########################################################################
########                        TASK 1                            ########
##########################################################################


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class NatureCNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_out):
        super().__init__()

        init_mid = lambda m: init(
                    m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                        nn.init.calculate_gain('relu')
                )
        init_last = lambda m: init(
                    m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0)
                )

        self.feature_extractor = nn.Sequential(
            init_mid(
                nn.Conv2d(n_input, 32, 8, stride=4)
            ),
            nn.ReLU(),

            init_mid(
                nn.Conv2d(32, 64, 4, stride=2)
            ),
            nn.ReLU(),

            init_mid(
                nn.Conv2d(64, 32, 3, stride=1)
            ),
            nn.ReLU(),
            Flatten()
        )

        self.head = nn.Sequential(
            init_mid(
                nn.Linear(32*7*7, n_hidden)
            ),
            nn.ReLU(),

            init_last(
                nn.Linear(n_hidden, n_out)
            )
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x



