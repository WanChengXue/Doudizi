from torch import nn
from torch import Tensor
import torch


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.state_conv1 = nn.Conv2d(
            8, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.state_bn1 = nn.BatchNorm2d(64)
        self.state_conv2 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.state_bn2 = nn.BatchNorm2d(64)
        self.state_conv3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.state_bn3 = nn.BatchNorm2d(64)
        self.state_conv4 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.state_bn4 = nn.BatchNorm2d(64)
        self.state_conv5 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.state_bn5 = nn.BatchNorm2d(64)

        self.action_conv1 = nn.Conv2d(
            2, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.action_bn1 = nn.BatchNorm2d(64)
        self.action_conv2 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.action_bn2 = nn.BatchNorm2d(64)
        self.action_conv3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.action_bn3 = nn.BatchNorm2d(64)
        self.action_conv4 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.action_bn4 = nn.BatchNorm2d(64)
        self.action_conv5 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.action_bn5 = nn.BatchNorm2d(64)

        self.activate = nn.LeakyReLU()

        # output
        self.out_conv1 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.out_bn1 = nn.BatchNorm2d(128)
        self.out_conv2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, padding_mode="circular"
        )
        self.out_bn2 = nn.BatchNorm2d(128)
        self.out_linear1 = nn.Linear(7680, 256)
        self.out_linear2 = nn.Linear(256, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        state: [batch_size, channel, row, col]
        action: [batch_size, channel, row, col]
        """
        # current state
        state_ = self.activate(self.state_bn1(self.state_conv1(state)))
        state_ = state_ + self.activate(self.state_bn2(self.state_conv2(state_)))
        state_ = state_ + self.activate(self.state_bn3(self.state_conv3(state_)))
        state_ = state_ + self.activate(self.state_bn4(self.state_conv4(state_)))
        state_ = state_ + self.activate(self.state_bn5(self.state_conv5(state_)))

        # chosen action
        action_ = self.activate(self.action_bn1(self.action_conv1(action)))
        action_ = action_ + self.activate(self.action_bn2(self.action_conv2(action_)))
        action_ = action_ + self.activate(self.action_bn3(self.action_conv3(action_)))
        action_ = action_ + self.activate(self.action_bn4(self.action_conv4(action_)))
        action_ = action_ + self.activate(self.action_bn5(self.action_conv5(action_)))

        output = torch.cat([state_, action_], dim=1)
        output = output + self.activate(self.out_bn1(self.out_conv1(output)))
        output = output + self.activate(self.out_bn2(self.out_conv2(output)))
        output = torch.flatten(output, 1)

        score = self.activate(self.out_linear1(output))
        score = self.out_linear2(score)

        return score
