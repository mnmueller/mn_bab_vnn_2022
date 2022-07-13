from __future__ import annotations

from typing import Any, Optional, Tuple

from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers.reshape import Reshape as concreteReshape
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class Reshape(concreteReshape, AbstractModule):
    def __init__(
        self,
        out_dim: Tuple[int, ...],
        input_dim: Tuple[int, ...],
    ):
        super(Reshape, self).__init__(out_dim)
        self.input_dim = input_dim
        # We assume no batch-dim
        self.output_dim = out_dim

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: concreteReshape, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Reshape:
        assert isinstance(module, concreteReshape)
        abstract_layer = cls(
            module.shape,
            input_dim,
        )
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
        if abstract_shape.uses_dependence_sets():
            assert False, "Not implemented - Reshape with dependence sets"
        else:
            assert isinstance(affine_form.coef, Tensor)
        new_coef = affine_form.coef.reshape(
            (*affine_form.coef.shape[:2], *self.input_dim)
        )
        new_bias = affine_form.bias
        return AffineForm(new_coef, new_bias)

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
    ) -> Tuple[Tensor, Tensor]:

        interval_lb, interval_ub = interval
        output_lb = interval_lb.reshape(self.output_dim)
        output_ub = interval_ub.reshape(self.output_dim)

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.view((abs_input.shape[0], *self.output_dim))
