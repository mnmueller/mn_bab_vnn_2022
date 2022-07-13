import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import dill  # type: ignore[import]
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_sig_base import SigBase
from src.abstract_layers.abstract_sigmoid import Sigmoid, d_sig, sig
from src.abstract_layers.abstract_tanh import Tanh, d_tanh, tanh
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.config import Config, Dtype, make_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import freeze_network, load_onnx_model
from src.utilities.output_property_form import OutputPropertyForm
from src.verification_instance import (
    generate_adv_output_str,
    get_sigmoid_gt_constraint,
    get_unet_gt_constraint,
)

TEMP_RUN_DIR = "run"


def generate_constraints(
    class_num: int, y: int
) -> List[List[List[Tuple[int, int, float]]]]:
    return [[[(y, i, 0)] for i in range(class_num) if i != y]]


def load_io_constraints(
    metadata: Dict[str, str], torch_dtype: torch.dtype, torch_device: torch.device
) -> Tuple[
    List[Tensor], List[Tuple[Tensor, Tensor]], List[List[List[Tuple[int, int, float]]]]
]:

    input_path = f"{TEMP_RUN_DIR}/inputs"
    inputs: List[Tensor] = []
    stack_input = torch.load(f"{input_path}/inputs.pt")
    inputs = list(torch.split(stack_input, 1, dim=0))
    inputs = [inp.squeeze(dim=0) for inp in inputs]

    input_region_path = f"{TEMP_RUN_DIR}/input_regions"
    input_regions: List[Tuple[Tensor, Tensor]] = []
    stack_lbs = torch.load(f"{input_region_path}/input_lbs.pt")
    stack_ubs = torch.load(f"{input_region_path}/input_ubs.pt")
    input_regions = [(lb, ub) for (lb, ub) in zip(stack_lbs, stack_ubs)]

    target_g_t_constraint_path = f"{TEMP_RUN_DIR}/io_constraints"

    with open(f"{target_g_t_constraint_path}/target_g_t_constraints.pkl", "rb") as file:
        target_g_t_constraints: List[List[List[Tuple[int, int, float]]]] = dill.load(
            file
        )

    return inputs, input_regions, target_g_t_constraints


def get_asnet(
    net_path: str, config: Config, device: torch.device
) -> Tuple[nn.Module, AbstractNetwork]:
    # Get bounds
    net_format = net_path.split(".")[-1]
    if net_format in ["onnx", "gz"]:
        net_seq, onnx_shape, inp_name = load_onnx_model(net_path)  # Like this for mypy
        net: nn.Module = net_seq
    else:
        assert False, f"No net loaded for net format: {net_format}."

    net.to(device)
    net.eval()
    freeze_network(net)

    if config.dtype == Dtype.float64:
        net = net.double()
    else:
        net = net.float()

    assert isinstance(net, nn.Sequential)
    as_net = AbstractNetwork.from_concrete_module(net, config.input_dim).to(device)
    freeze_network(as_net)
    return net, as_net


def load_meta_data(metadata_path: str) -> Dict[str, str]:
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    metadata: Dict[str, str] = {}
    for line in lines:
        k, v = line.split(":")[:2]
        k = k.strip()
        v = v.strip()
        metadata[k] = v

    return metadata


def run_instance(
    benchmark: str, net_path: str, spec_path: str, res_path: str, timeout: int
) -> None:

    metadata_path = f"{TEMP_RUN_DIR}/metadata.txt"
    config_path = f"{TEMP_RUN_DIR}/config.json"
    abs_network_path = f"{TEMP_RUN_DIR}/abs_network.onnx"

    # 1. Check metadata (Shallow)
    metadata = load_meta_data(metadata_path)
    assert (
        metadata["benchmark"] == benchmark
    ), f"Benchmarks don't match {metadata['benchmark']} {benchmark}"
    assert (
        metadata["network_path"] == net_path
    ), f"Networks don't match {metadata['network_path']} {net_path}"
    assert (
        metadata["spec_path"] == spec_path
    ), f"Specs don't match {metadata['spec_path']} {spec_path}"

    # Get Config
    config = get_config_from_json(config_path)
    parsed_config = make_config(**config)
    seed_everything(parsed_config.random_seed)
    # Set timeout
    parsed_config.timeout = timeout

    if torch.cuda.is_available() and parsed_config.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if parsed_config.dtype == Dtype.float64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32

    # Get data
    (inputs, input_regions, target_g_t_constraints) = load_io_constraints(
        metadata, torch_dtype=dtype, torch_device=device
    )

    network: AbstractNetwork = torch.load(abs_network_path)
    network.eval()
    network.to(dtype).to(device)
    network.set_activation_layers()

    check_sigmoid_tanh_tangent(network)

    sigmoid_encode_property: bool = False
    if isinstance(network.layers[-1], Sigmoid):
        network.layers = network.layers[:-1]
        network.set_activation_layers()
        sigmoid_encode_property = True

    # Get verifier
    verifier = MNBaBVerifier(network, device, parsed_config.verifier)

    start_time = time.time()
    for input_point, (input_lb, input_ub), gt_constraint in zip(
        inputs, input_regions, target_g_t_constraints
    ):
        if input_lb.dim() == len(parsed_config.input_dim):  # No batch dimension
            input_lb = input_lb.unsqueeze(0)
            input_ub = input_ub.unsqueeze(0)
            input_point = input_point.unsqueeze(0)
        assert tuple(input_lb.shape[1:]) == parsed_config.input_dim

        if sigmoid_encode_property:
            gt_constraint = get_sigmoid_gt_constraint(gt_constraint)

        if "unet" in config.benchmark_instances_path:
            gt_constraint, gt_target = get_unet_gt_constraint(
                input_point, (input_lb, input_ub), gt_constraint
            )

        assert isinstance(network, AbstractNetwork)
        out_prop_form = OutputPropertyForm.create_from_properties(
            gt_constraint,
            None,
            parsed_config.verifier.outer.use_disj_adapter,
            network.output_dim[-1],
            device,
            torch.get_default_dtype(),
        )

        if "unet" in config.benchmark_instances_path:

            assert gt_target
            (is_verified, _, _, _, _,) = verifier.verify_unet_via_config(
                0,
                input_point,
                input_lb,
                input_ub,
                out_prop_form,
                verification_target=gt_target,
                timeout=config.timeout + start_time,
            )

        else:

            if out_prop_form.disjunction_adapter is not None:
                verifier.append_out_adapter(
                    out_prop_form.disjunction_adapter,
                    device,
                    torch.get_default_dtype(),
                )

            (
                is_verified,
                adv_example,
                lower_idx,
                lower_bound_tmp,
                upper_bound_tmp,
            ) = verifier.verify_via_config(
                0,
                input_point,
                input_lb,
                input_ub,
                out_prop_form,
                timeout=parsed_config.timeout + start_time,
            )

            if out_prop_form.disjunction_adapter is not None:
                verifier.remove_out_adapter()

            if adv_example is not None:
                assert not is_verified
                break
            if not is_verified:
                break

    total_time = time.time() - start_time

    Path("/".join(res_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f:
        if is_verified:
            f.write("unsat\n")
        elif adv_example is not None:
            f.write("sat\n")
            counter_example = generate_adv_output_str(
                adv_example[0], network, input_lb.shape, device, dtype
            )
            f.write(counter_example)
        elif total_time >= timeout:
            f.write("timeout\n")
        else:
            f.write("unknown\n")


def check_sigmoid_tanh_tangent(network: AbstractNetwork) -> None:
    has_sig_layer = False
    has_tanh_layer = False
    for tag, layer in network.layer_id_to_layer.items():
        if isinstance(layer, Sigmoid):
            has_sig_layer = True
        if isinstance(layer, Tanh):
            has_tanh_layer = True

    if has_sig_layer:
        (
            Sigmoid.intersection_points,
            Sigmoid.tangent_points,
            Sigmoid.step_size,
            Sigmoid.max_x,
        ) = SigBase._compute_bound_to_tangent_point(sig, d_sig)

    if has_tanh_layer:
        (
            Tanh.intersection_points,
            Tanh.tangent_points,
            Tanh.step_size,
            Tanh.max_x,
        ) = SigBase._compute_bound_to_tangent_point(tanh, d_tanh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare verification instances on the vnn22 datasets. Simply provide the corresponding nets and specs"
    )
    parser.add_argument(
        "-b", "--benchmark", type=str, help="The benchmark id", required=True
    )
    parser.add_argument(
        "-n",
        "--netname",
        type=str,
        help="The network path",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--vnnlib_spec",
        type=str,
        help="The vnnlib spec path",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--results_path",
        type=str,
        help="",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        help="",
        required=True,
    )

    args = parser.parse_args()

    run_instance(
        args.benchmark,
        args.netname,
        args.vnnlib_spec,
        args.results_path,
        int(args.timeout),
    )
