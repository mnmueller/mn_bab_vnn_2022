from __future__ import annotations

from math import floor
from typing import Any, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets
from src.utilities.general import get_neg_pos_comp
from src.verification_subproblem import SubproblemState


class Conv2d(nn.Conv2d, AbstractModule):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]  # Tuple[int, int] ?
    input_dim: Tuple[int, ...]
    stride: Tuple[int, ...]  # Tuple[int, int] ?
    padding: Tuple[int, ...]  # type: ignore[assignment] # checked at runtime below (Tuple[int, int] ?)
    dilation: Tuple[int, ...]  # Tuple[int, int] ?
    groups: int
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        input_dim: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(Conv2d, self).__init__(  # type: ignore # mypy issue 4335
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.input_dim = input_dim

        assert not isinstance(self.padding, str)

        output_height = floor(
            (
                input_dim[1]
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        output_width = floor(
            (
                input_dim[2]
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )
        self.output_dim = (out_channels, output_height, output_width)
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Conv2d, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Conv2d:
        assert isinstance(module, nn.Conv2d)
        assert len(module.kernel_size) == 2
        assert len(module.stride) == 2
        assert len(module.padding) == 2
        assert len(module.dilation) == 2
        assert not isinstance(module.padding, str)
        abstract_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,  # type: ignore[arg-type]
            input_dim,
            module.stride,  # type: ignore[arg-type]
            module.padding,  # type: ignore[arg-type]
            module.dilation,  # type: ignore[arg-type]
            module.groups,
            module.bias is not None,
        )
        abstract_layer.weight.data = module.weight.data
        if module.bias is not None:
            assert abstract_layer.bias is not None
            abstract_layer.bias.data = module.bias.data
        return abstract_layer

    def backsubstitute(
        self, config: BacksubstitutionConfig, abstract_shape: MN_BaB_Shape
    ) -> MN_BaB_Shape:

        new_lb_form = self._backsub_affine_form(abstract_shape.lb, abstract_shape)
        new_ub_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            new_ub_form = self._backsub_affine_form(abstract_shape.ub, abstract_shape)

        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def _backsub_affine_form(
        self, affine_form: AffineForm, abstract_shape: MN_BaB_Shape
    ) -> AffineForm:
        new_coef: Union[Tensor, DependenceSets]

        if abstract_shape.uses_dependence_sets():
            symmetric_stride = self.stride[0] == self.stride[1]
            symmetric_padding = self.padding[0] == self.padding[1]
            dilation_one = (
                self.dilation[0] == self.dilation[1] == 1 and self.dilation[0] == 1
            )
            group_one = self.groups == 1
            dependence_sets_assumptions = (
                symmetric_stride and symmetric_padding and dilation_one and group_one
            )
            assert dependence_sets_assumptions, "Dependence set assumptions violated."

            def backsubstitute_coef_and_bias(
                coef: DependenceSets, bias: Tensor
            ) -> Tuple[DependenceSets, Tensor]:
                assert isinstance(affine_form.coef, DependenceSets)
                new_bias = bias + (
                    0
                    if self.bias is None
                    else (
                        DependenceSets.unfold_to(
                            self.bias.unsqueeze(0)
                            .unsqueeze(2)
                            .unsqueeze(3)
                            .expand(bias.shape[0], *self.output_dim),
                            affine_form.coef,
                        )
                        * affine_form.coef.sets
                    ).sum((2, 3, 4))
                )
                # [B*C*HW, c, d, d] -> [B*C*HW, c', d', d']
                new_coef_sets = F.conv_transpose2d(
                    coef.sets.flatten(end_dim=1), self.weight, stride=self.stride
                )
                assert not isinstance(self.padding, str)
                new_coef = DependenceSets(
                    new_coef_sets.view(*coef.sets.shape[:2], *new_coef_sets.shape[1:]),
                    coef.spatial_idxs,
                    coef.input_dim,
                    coef.cstride * self.stride[0],
                    coef.cpadding * self.stride[0] + self.padding[0],
                )
                return new_coef, new_bias

            assert isinstance(affine_form.coef, DependenceSets)
            new_coef, new_bias = backsubstitute_coef_and_bias(
                affine_form.coef, affine_form.bias
            )
            new_coef.handle_padding(self.input_dim)
        else:
            assert isinstance(affine_form.coef, Tensor)
            assert not isinstance(self.padding, str)

            kernel_wh = self.weight.shape[-2:]
            w_padding = (
                self.input_dim[1]
                + 2 * self.padding[0]
                - 1
                - self.dilation[0] * (kernel_wh[0] - 1)
            ) % self.stride[0]
            h_padding = (
                self.input_dim[2]
                + 2 * self.padding[1]
                - 1
                - self.dilation[1] * (kernel_wh[1] - 1)
            ) % self.stride[1]
            output_padding = (w_padding, h_padding)

            sz = affine_form.coef.shape

            # process reference
            new_bias = affine_form.bias + (
                0
                if self.bias is None
                else (affine_form.coef.sum((3, 4)) * self.bias).sum(2)
            )

            new_coef = F.conv_transpose2d(
                affine_form.coef.view((sz[0] * sz[1], *sz[2:])),
                self.weight,
                None,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

            # F.pad(new_x_l_coef, (0, 0, w_padding, h_padding), "constant", 0)
            assert isinstance(new_coef, Tensor)
            new_coef = new_coef.view((sz[0], sz[1], *new_coef.shape[1:]))

        return AffineForm(new_coef, new_bias)

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        neg_kernel, pos_kernel = get_neg_pos_comp(self.weight)

        def conv_with_kernel_and_bias(
            input: Tensor, kernel: Tensor, bias: Optional[Tensor]
        ) -> Tensor:
            return F.conv2d(
                input=input,
                weight=kernel,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        output_lb = conv_with_kernel_and_bias(
            interval_lb, pos_kernel, self.bias
        ) + conv_with_kernel_and_bias(interval_ub, neg_kernel, None)
        output_ub = conv_with_kernel_and_bias(
            interval_ub, pos_kernel, self.bias
        ) + conv_with_kernel_and_bias(interval_lb, neg_kernel, None)

        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        assert all([x == self.stride[0] for x in self.stride])
        assert all([x == self.padding[0] for x in self.padding])
        assert all([x == self.dilation[0] for x in self.dilation])

        return abs_input.conv2d(
            self.weight,
            self.bias,
            self.stride[0],
            self.padding[0],
            self.dilation[0],
            self.groups,
        )
