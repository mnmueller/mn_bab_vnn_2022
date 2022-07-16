import shutil
import time
from pathlib import Path

import torch
from torch import nn

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_network import AbstractNetwork
from src.milp_network import MILPNetwork
from src.state.tags import layer_tag
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import load_onnx_model
from tests.test_util import (
    get_deep_poly_bounds,
    toy_pad_net,
    toy_permute_net,
    toy_unbinary_net,
)


class TestPadPermute:
    """
    We test our Padding and Permutation layer.
    """

    def test_padding(self) -> None:

        net_pt = toy_pad_net()
        shape = (1, 16, 16)
        onnx_shape = (1, 1, 16, 16)
        eps = 2 / 255

        try:
            temp_dir = "tests/temp"
            net_pref = "pad_test"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            x = torch.rand(onnx_shape)

            torch.onnx.export(
                net_pt,
                x,
                onnx_path,
                export_params=True,
                training=0,
                do_constant_folding=False,
                verbose=False,
                input_names=["input.1"],
                output_names=["output"],
            )

            o2p_net, _, in_name = load_onnx_model(onnx_path)
            o2p_net.eval()
            # Compare results

            for i in range(10):
                x = torch.rand(onnx_shape)

                out_pt_net = net_pt(x)
                out_o2p_net = o2p_net(x)
                assert torch.isclose(out_pt_net, out_o2p_net).all()

            # Get abstract net
            assert isinstance(o2p_net, nn.Sequential)
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, shape)

            input = torch.rand(onnx_shape)
            input_lb = input - eps
            input_ub = input + eps
            (
                output_lb_without_alpha,
                output_ub_without_alpha,
            ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

            x = input
            x_out = abs_net(x)

            in_zono = HybridZonotope.construct_from_bounds(
                input_lb, input_ub, domain="zono"
            )
            out_zono = abs_net.propagate_abstract_element(in_zono)
            assert out_zono.shape == x_out.shape
            assert out_zono.may_contain_point(x_out)

            in_dpf = DeepPoly_f.construct_from_bounds(input_lb, input_ub, domain="DPF")
            out_dpf = abs_net.propagate_abstract_element(in_dpf)
            assert out_dpf.shape == x_out.shape
            assert out_dpf.may_contain_point(x_out)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_padding_milp(self) -> None:

        seed_everything(1)
        net_pt = toy_pad_net()
        shape = (1, 16, 16)
        onnx_shape = (1, 1, 16, 16)
        eps = 2 / 255

        try:
            temp_dir = "tests/temp"
            net_pref = "pad_test"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            x = torch.rand(onnx_shape)

            torch.onnx.export(
                net_pt,
                x,
                onnx_path,
                export_params=True,
                training=0,
                do_constant_folding=False,
                verbose=False,
                input_names=["input.1"],
                output_names=["output"],
            )

            o2p_net, _, in_name = load_onnx_model(onnx_path)
            o2p_net.eval()

            # Get abstract net
            assert isinstance(o2p_net, nn.Sequential)
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, shape)

            input = torch.rand(onnx_shape)
            input_lb = input - eps
            input_ub = input + eps

            milp_model = MILPNetwork.build_model_from_abstract_net(
                input, input_lb, input_ub, abs_net
            )
            for i, layer in enumerate(milp_model.net.layers):
                lbs, ubs = milp_model.get_network_bounds_at_layer_multi(
                    layer_tag(layer),
                    timeout_per_instance=20.0,
                    timeout_total=400.0,
                    timeout=time.time() + 400,
                )
                print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
                assert (
                    lbs >= milp_model.net.layers[i].output_bounds[0].flatten()
                ).all()
                assert (
                    ubs <= milp_model.net.layers[i].output_bounds[1].flatten()
                ).all()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_permutation(self) -> None:

        net_pt = toy_permute_net()
        shape = (1, 4, 8)
        onnx_shape = (1, 1, 4, 8)
        eps = 2 / 255

        try:
            temp_dir = "tests/temp"
            net_pref = "pad_test"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            x = torch.rand(onnx_shape)

            torch.onnx.export(
                net_pt,
                x,
                onnx_path,
                export_params=True,
                training=0,
                do_constant_folding=False,
                verbose=False,
                input_names=["input.1"],
                output_names=["output"],
            )

            o2p_net, _, in_name = load_onnx_model(onnx_path)
            o2p_net.eval()
            # Compare results

            for i in range(10):
                x = torch.rand(onnx_shape)

                out_pt_net = net_pt(x)
                out_o2p_net = o2p_net(x)
                assert torch.isclose(out_pt_net, out_o2p_net).all()

            # Get abstract net
            assert isinstance(o2p_net, nn.Sequential)
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, shape)

            input = torch.rand(onnx_shape)
            input_lb = input - eps
            input_ub = input + eps
            (
                output_lb_without_alpha,
                output_ub_without_alpha,
            ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_permutation_milp(self) -> None:

        net_pt = toy_permute_net()
        shape = (1, 4, 8)
        onnx_shape = (1, 1, 4, 8)
        eps = 2 / 255

        try:
            temp_dir = "tests/temp"
            net_pref = "pad_test"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            x = torch.rand(onnx_shape)

            torch.onnx.export(
                net_pt,
                x,
                onnx_path,
                export_params=True,
                training=0,
                do_constant_folding=False,
                verbose=False,
                input_names=["input.1"],
                output_names=["output"],
            )

            o2p_net, _, in_name = load_onnx_model(onnx_path)
            o2p_net.eval()

            # Get abstract net
            assert isinstance(o2p_net, nn.Sequential)
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, shape)

            input = torch.rand(onnx_shape)
            input_lb = input - eps
            input_ub = input + eps

            milp_model = MILPNetwork.build_model_from_abstract_net(
                input, input_lb, input_ub, abs_net
            )
            for i, layer in enumerate(milp_model.net.layers):
                lbs, ubs = milp_model.get_network_bounds_at_layer_multi(
                    layer_tag(layer),
                    timeout_per_instance=20,
                    timeout_total=400,
                    timeout=time.time() + 400,
                )
                print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
                assert (
                    lbs >= milp_model.net.layers[i].output_bounds[0].flatten()
                ).all()
                assert (
                    ubs <= milp_model.net.layers[i].output_bounds[1].flatten()
                ).all()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_unbinary_op(self) -> None:
        seed_everything(0)
        network = toy_unbinary_net()
        input = torch.ones(size=(1, 1, 4, 4))
        eps = 0.5
        num_eps = 1e-6
        input_lb, input_ub = input - eps, input + eps
        q = network(input)
        milp_model = MILPNetwork.build_model_from_abstract_net(
            input, input_lb, input_ub, network
        )
        lbs, ubs = milp_model.get_network_output_bounds()
        (
            output_lb_deep_poly,
            output_ub_deep_poly,
        ) = get_deep_poly_bounds(network, input_lb, input_ub)
        assert (q >= lbs).all()
        assert (q <= ubs).all()
        assert (
            output_lb_deep_poly < lbs + num_eps
        ).all(), (
            f"Found violation lb {torch.max(output_lb_deep_poly - (lbs + num_eps))}"
        )

    def test_unbinary_op_milp(self) -> None:
        net_pt = toy_unbinary_net()
        eps = 2 / 255
        shape = (1, 4, 4)
        onnx_shape = (1, 1, 4, 4)

        try:
            temp_dir = "tests/temp"
            net_pref = "pad_test"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            x = torch.rand(onnx_shape)

            torch.onnx.export(
                net_pt,
                x,
                onnx_path,
                export_params=True,
                training=0,
                do_constant_folding=False,
                verbose=False,
                input_names=["input.1"],
                output_names=["output"],
            )

            o2p_net, _, in_name = load_onnx_model(onnx_path)
            o2p_net.eval()

            # Get abstract net
            assert isinstance(o2p_net, nn.Sequential)
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, shape)

            input = torch.rand(onnx_shape)
            input_lb = input - eps
            input_ub = input + eps

            milp_model = MILPNetwork.build_model_from_abstract_net(
                input, input_lb, input_ub, abs_net
            )
            for i, layer in enumerate(milp_model.net.layers):
                lbs, ubs = milp_model.get_network_bounds_at_layer_multi(
                    layer_tag(layer),
                    timeout_per_instance=20,
                    timeout_total=400,
                    timeout=time.time() + 400,
                )
                print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
                assert (
                    lbs >= milp_model.net.layers[i].output_bounds[0].flatten()
                ).all()
                assert (
                    ubs <= milp_model.net.layers[i].output_bounds[1].flatten()
                ).all()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    t = TestPadPermute()
    # t.test_padding()
    t.test_padding_milp()
    # t.test_permutation()
    # t.test_permutation_milp()
    # t.test_unbinary_op_milp()
    # t.test_max_avg_pooling_milp()
