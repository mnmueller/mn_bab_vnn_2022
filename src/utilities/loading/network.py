import gzip
import typing
from os.path import exists
from typing import List, Optional, Sequence, Tuple, Type

import onnx  # type: ignore[import]
import torch
from bunch import Bunch  # type: ignore[import]
from torch import nn as nn

from src.concrete_layers.basic_block import BasicBlock
from src.utilities.onnx_loader import ConvertModel


def mnist_conv_tiny() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=8 * 2 * 2, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
        ]
    )


def mnist_conv_small() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 5 * 5, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )


def mnist_conv_sigmoid_small() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 5 * 5, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )


def mnist_conv_big() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


def mnist_conv_super() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 18 * 18, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


def mnist_a_b(a: int, b: int) -> nn.Sequential:
    layers = [nn.Linear(28 * 28, b), nn.ReLU()]
    for __ in range(a - 1):
        layers += [
            nn.Linear(b, b),
            nn.ReLU(),
        ]
    layers += [nn.Linear(b, 10), nn.ReLU()]
    return nn.Sequential(*layers)


def mnist_sig_a_b(a: int, b: int) -> nn.Sequential:
    layers = [nn.Linear(28 * 28, b), nn.Sigmoid()]
    for __ in range(a - 1):
        layers += [
            nn.Linear(b, b),
            nn.Sigmoid(),
        ]
    layers += [nn.Linear(b, 10), nn.Sigmoid()]
    return nn.Sequential(*layers)


def mnist_vnncomp_a_b(a: int, b: int) -> nn.Sequential:
    layers = [nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(28 * 28, b), nn.ReLU()]
    for __ in range(a - 1):
        layers += [
            nn.Linear(b, b),
            nn.ReLU(),
        ]
    layers += [nn.Linear(b, 10)]
    return nn.Sequential(*layers)


def cifar10_conv_small() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 6 * 6, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )


def cifar10_cnn_A() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 8 * 8, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )


def cifar10_base() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cifar10_wide() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cifar10_deep() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(8 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cifar10_2_255_simplified() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128 * 8 * 8, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cifar10_8_255_simplified() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128 * 8 * 8, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cifar10_conv_big() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


def getShapeConv(
    in_shape: Tuple[int, int, int],
    conv_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int]:
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


class ResNet(nn.Sequential):
    def __init__(
        self,
        block: Type[BasicBlock],
        in_ch: int = 3,
        num_stages: int = 1,
        num_blocks: int = 2,
        num_classes: int = 10,
        in_planes: int = 64,
        bn: bool = True,
        last_layer: str = "avg",
        in_dim: int = 32,
        stride: Optional[Sequence[int]] = None,
    ):
        layers: List[nn.Module] = []
        self.in_planes = in_planes
        if stride is None:
            stride = (num_stages + 1) * [2]

        layers.append(
            nn.Conv2d(
                in_ch,
                self.in_planes,
                kernel_size=3,
                stride=stride[0],
                padding=1,
                bias=not bn,
            )
        )

        _, _, in_dim = getShapeConv(
            (in_ch, in_dim, in_dim), (self.in_planes, 3, 3), stride=stride[0], padding=1
        )

        if bn:
            layers.append(nn.BatchNorm2d(self.in_planes))

        layers.append(nn.ReLU())

        for s in stride[1:]:
            block_layers, in_dim = self._make_layer(
                block,
                self.in_planes * 2,
                num_blocks,
                stride=s,
                bn=bn,
                kernel=3,
                in_dim=in_dim,
            )
            layers.append(block_layers)

        if last_layer == "avg":
            layers.append(nn.AvgPool2d(4))
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    self.in_planes * (in_dim // 4) ** 2 * block.expansion, num_classes
                )
            )
        elif last_layer == "dense":
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(self.in_planes * block.expansion * in_dim**2, 100)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Linear(100, num_classes))
        else:
            exit("last_layer type not supported!")

        super(ResNet, self).__init__(*layers)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_layers: int,
        stride: int,
        bn: bool,
        kernel: int,
        in_dim: int,
    ) -> Tuple[nn.Sequential, int]:
        strides = [stride] + [1] * (num_layers - 1)
        cur_dim: int = in_dim
        layers: List[nn.Module] = []
        for stride in strides:
            layer = block(self.in_planes, planes, stride, bn, kernel, in_dim=cur_dim)
            layers.append(layer)
            cur_dim = layer.out_dim
            layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers), cur_dim


def resnet2b(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock, num_stages=1, num_blocks=2, in_planes=8, bn=bn, last_layer="dense"
    )


def resnet2b2(bn: bool = False, in_ch: int = 3, in_dim: int = 32) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=in_ch,
        num_stages=2,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2],
    )


def resnet4b(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock, num_stages=2, num_blocks=2, in_planes=8, bn=bn, last_layer="dense"
    )


def resnet4b1(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=4,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[1, 1, 2, 2, 2],
    )


def resnet4b2(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=4,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 1, 1],
    )


def resnet3b2(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=3,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 2],
    )


def resnet9b(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=3,
        num_blocks=3,
        in_planes=16,
        bn=bn,
        last_layer="dense",
    )


def freeze_network(network: nn.Module) -> None:
    for param in network.parameters():
        param.requires_grad = False


def load_net_from(config: Bunch) -> nn.Module:
    path = config.network_path
    try:
        n_layers = config.n_layers
        n_neurons_per_layer = config.n_neurons_per_layer
    except AttributeError:
        n_layers = None
        n_neurons_per_layer = None
    return load_net(path, n_layers, n_neurons_per_layer)


def load_net(  # noqa: C901
    path: str, n_layers: Optional[int], n_neurons_per_layer: Optional[int]
) -> nn.Module:
    if path.split(".")[-1] in ["onnx", "gz"]:
        return load_onnx_model(path)[0]
    elif "mnist_sig" in path and "flattened" in path:
        assert n_layers is not None and n_neurons_per_layer is not None
        original_network = mnist_sig_a_b(n_layers, n_neurons_per_layer)
    elif "mnist" in path and "flattened" in path:
        assert n_layers is not None and n_neurons_per_layer is not None
        original_network = mnist_a_b(n_layers, n_neurons_per_layer)
    elif "mnist-net" in path:
        assert n_layers is not None and n_neurons_per_layer is not None
        original_network = mnist_vnncomp_a_b(n_layers, n_neurons_per_layer)
    elif "mnist_convSmallRELU__Point" in path:
        original_network = mnist_conv_small()
    elif "mnist_SIGMOID" in path:
        original_network = mnist_conv_sigmoid_small()
    elif "mnist_convBigRELU__DiffAI" in path:
        original_network = mnist_conv_big()
    elif "mnist_convSuperRELU__DiffAI" in path:
        original_network = mnist_conv_super()
    elif "cifar10_convSmallRELU__PGDK" in path:
        original_network = cifar10_conv_small()
    elif "cifar10_CNN_A_CIFAR_MIX" in path:
        original_network = cifar10_cnn_A()
    elif "cifar_base_kw" in path:
        original_network = cifar10_base()
    elif "cifar_wide_kw" in path:
        original_network = cifar10_wide()
    elif "cifar_deep_kw" in path:
        original_network = cifar10_deep()
    elif "cifar10_2_255_simplified" in path:
        original_network = cifar10_2_255_simplified()
    elif "cifar10_8_255_simplified" in path:
        original_network = cifar10_8_255_simplified()
    elif "cifar10_convBigRELU__PGD" in path:
        original_network = cifar10_conv_big()
    elif "resnet_2b2" in path:
        original_network = resnet2b2(bn="bn" in path)
    elif "resnet_2b" in path:
        original_network = resnet2b()
    elif "resnet_3b2" in path:
        original_network = resnet3b2(bn="bn" in path)
    elif "resnet_4b1" in path:
        original_network = resnet4b1(bn="bn" in path)
    elif "resnet_4b2" in path:
        original_network = resnet4b2(bn="bn" in path)
    elif "resnet_4b" in path:
        original_network = resnet4b()
    elif "resnet_9b_bn" in path:
        original_network = resnet9b(bn=True)
    else:
        raise NotImplementedError(
            "The network specified in the configuration, could not be loaded."
        )
    state_dict = torch.load(path)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    original_network.load_state_dict(state_dict)
    freeze_network(original_network)

    return original_network


def load_onnx_model(path: str) -> Tuple[nn.Sequential, Tuple[int, ...], str]:
    onnx_model = load_onnx(path)
    return load_onnx_from_proto(onnx_model, path)


def load_onnx_from_proto(
    onnx_model: onnx.ModelProto, path: Optional[str] = None
) -> Tuple[nn.Sequential, Tuple[int, ...], str]:

    onnx_input_dims = onnx_model.graph.input[-1].type.tensor_type.shape.dim
    inp_name = onnx_model.graph.input[-1].name
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    pytorch_model = ConvertModel(onnx_model)

    if path is not None and "unet" in path:
        pytorch_structured = pytorch_model.forward_trace_to_graph_unet()
        softmax_idx = [
            i for (i, layer) in enumerate(pytorch_structured) if "Softmax" in str(layer)
        ][0]
        pytorch_structured = pytorch_structured[: softmax_idx - 1]
        pytorch_structured.append(nn.Flatten())
    elif path is not None and "vgg16-7" in path:
        onnx_shape = (3, 224, 224)
        pytorch_structured = pytorch_model.forward_trace_to_graph()
    else:
        pytorch_structured = pytorch_model.forward_trace_to_graph()

    return pytorch_structured, onnx_shape, inp_name


@typing.no_type_check
def load_onnx(path: str):
    # The official benchmark repo has all networks with the wrong ending
    if not exists(path) and not path.endswith(".gz"):
        path = path + ".gz"
    if path.endswith(".gz"):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model
