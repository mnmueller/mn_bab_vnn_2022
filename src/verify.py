import csv
import time

from comet_ml import Experiment  # type: ignore[import]

import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.config import make_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, load_net

if __name__ == "__main__":
    args = get_args()
    config_file = get_config_from_json(args.config)
    config = make_config(**config_file)
    seed_everything(config.random_seed)

    experiment_logger = Experiment(**config.logger.comet_options)
    experiment_logger.set_name(config.experiment_name)
    experiment_logger.log_parameters(config_file)

    if torch.cuda.is_available() and config.use_gpu:
        device = torch.device("cuda")
        experiment_logger.log_text("Using gpu")
        experiment_logger.log_text(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        experiment_logger.log_text("Using cpu")

    original_network = load_net(**config.network.load_params())
    original_network.to(device)
    original_network.eval()
    assert isinstance(original_network, nn.Sequential)
    network = AbstractNetwork.from_concrete_module(
        original_network, config.input_dim
    ).to(device)
    freeze_network(network)

    verifier = MNBaBVerifier(network, device, config.verifier)

    test_file = open(config.test_data_path, "r")
    test_instances = csv.reader(test_file, delimiter=",")

    total_start_time = time.time()
    running_total_time = 0.0
    n_correct = 0
    n_verified = 0
    n_disproved = 0
    for i, (label, *pixel_values) in enumerate(test_instances):
        if i < args.test_from:
            continue
        if args.test_num > 0 and i - args.test_from >= args.test_num:
            break
        start_time = time.time()
        print("Verifying instance number:", i)
        input, input_lb, input_ub = transform_and_bound(pixel_values, config, device)

        pred_label = torch.argmax(original_network(input)).item()
        if pred_label != int(label):
            print("Network fails on test image, skipping.\n")
            continue
        else:
            n_correct += 1
        verified, disproved = verifier.verify(
            i, input, input_lb, input_ub, int(label), config.timeout
        )
        if verified:
            n_verified += 1
            print("Verified instance: ", i)
        elif disproved:
            n_disproved += 1
            print("Disproved instance: ", i)
        else:
            print("Unable to verify instance: ", i)
        iteration_time = time.time() - start_time
        running_total_time += iteration_time
        print("Iteration time: ", iteration_time)
        print("Running average verification time:", running_total_time / n_correct)
        print("Verified", n_verified, "out of", n_correct)
        print("Disproved", n_disproved, "out of", n_correct - n_verified)
        print()
    if not experiment_logger.disabled:
        verifier.bab.log_info(experiment_logger)
    print("Total time: ", time.time() - total_start_time)
