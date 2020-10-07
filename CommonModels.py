import torch
from torchvision.models import resnet18
import typing as _typing
import math

class FcNet(torch.nn.Module):
    """Simple fully connected network
    """
    def __init__(self, dim_input: int, dim_output: _typing.Optional[int] = None, num_hidden_units: _typing.List[int] = (32, 32)) -> None:
        """
        Args:

        """
        super(FcNet, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_hidden_units = num_hidden_units

        self.fc_net = self.construct_network()

    def construct_network(self):
        """
        """
        net = torch.nn.Sequential()
        net.add_module(
            name='layer0',
            module=torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_input, out_features=self.num_hidden_units[0]),
                torch.nn.ReLU()
            )
        )

        for i in range(1, len(self.num_hidden_units)):
            net.add_module(
                name='layer{0:d}'.format(i),
                module=torch.nn.Sequential(
                    torch.nn.Linear(in_features=self.num_hidden_units[i - 1], out_features=self.num_hidden_units[i]),
                    torch.nn.ReLU()
                )
            )
        
        net.add_module(
            name='classifier',
            module=torch.nn.Linear(in_features=self.num_hidden_units[-1], out_features=self.dim_output) if self.dim_output is not None \
                else torch.nn.Identity()
        )

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.fc_net(x)

class CNN(torch.nn.Module):
    """A simple convolutional module networks
    """
    def __init__(self, dim_output: _typing.Optional[int] = None, image_size: _typing.Tuple[int, int, int] = (3, 84, 84), bn_affine: bool = False) -> None:
        """Initialize an instance

        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
            image_size: a 3-d tuple consisting of (nc, iH, iW)

        """
        super(CNN, self).__init__()

        self.dim_output = dim_output
        self.image_size = image_size
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.num_channels = (32, 32, 32, 32)
        self.bn_affine = bn_affine

        self.nz = self.get_flatten_feature_dimension()

        self.cnn = self.construct_network()

    def get_flatten_feature_dimension(self):
        """Get the dimension of feature at the last convolutional layer"""
        _, iH, iW = self.image_size
        for _ in range(len(self.num_channels)):
            iH = (iH + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1
            iH = math.floor(iH)

            iW = (iW + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1
            iW = math.floor(iW)

        return iH * iW
    
    def construct_network(self) -> torch.nn.Module:
        """Construct the network

        """
        net = torch.nn.Sequential()
        net.add_module(
            name='layer0',
            module=torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.image_size[0],
                    out_channels=self.num_channels[0],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=not self.bn_affine
                ),
                torch.nn.BatchNorm2d(
                    num_features=self.num_channels[0],
                    momentum=1,
                    track_running_stats=False,
                    affine=self.bn_affine
                ),
                torch.nn.ReLU()
            )
        )

        for i in range(1, len(self.num_channels)):
            net.add_module(
                name='layer{0:d}'.format(i),
                module=torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=self.num_channels[i - 1],
                        out_channels=self.num_channels[i],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        bias=not self.bn_affine
                    ),
                    torch.nn.BatchNorm2d(
                        num_features=self.num_channels[i],
                        momentum=1,
                        track_running_stats=False,
                        affine=self.bn_affine
                    ),
                    torch.nn.ReLU()
                )
            )

        net.add_module(name='Flatten', module=torch.nn.Flatten(start_dim=1, end_dim=-1))

        if self.dim_output is None:
            clf = torch.nn.Identity()
        else:
            clf = torch.nn.Linear(
                in_features=self.nz * self.num_channels[-1],
                out_features=self.dim_output
            )

        net.add_module(name='classifier', module=clf)

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.cnn(x)

class ResNet18(torch.nn.Module):
    """A modified version of ResNet-18 that suits meta-learning"""
    def __init__(self, input_channel: int = 3, dim_output: _typing.Optional[int] = None, bn_affine: bool = False) -> None:
        """
        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
        """
        super(ResNet18, self).__init__()

        self.input_channel = input_channel
        self.dim_output = dim_output
        self.net = self.modified_resnet18()
        self.bn_affine = bn_affine

    def modified_resnet18(self):
        """
        """
        net = resnet18(pretrained=False)

        # modify the resnet to suit the data
        # change first conv2d from 7-by-7 kernel size to 3-by-3
        net.conv1 = torch.nn.Conv2d(
            in_channels=self.input_channel,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(3, 3),
            bias=not self.bn_affine
        )

        # update batch norm for meta-learning by setting momentum to 1
        net.bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[0].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[0].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[1].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[1].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[0].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].downsample[1] = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[1].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[1].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[0].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].downsample[1] = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[1].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[1].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[0].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].downsample[1] = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[1].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[1].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        # last layer
        if self.dim_output is not None:
            net.fc = torch.nn.Linear(in_features=512, out_features=self.dim_output)
        else:
            net.fc = torch.nn.Identity()

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.net(x)
