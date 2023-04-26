"""
Latent ODEs for Irregularly-Sampled Time Series
Author: Yulia Rubanova
"""

import torch
from torch.distributions import uniform
from torch.utils.data import DataLoader

from .utils import inf_generator, split_and_subsample_batch, split_train_test

#####################################################################################################


def sine(trajectories_to_sample, device):
    t_end = 20.0
    t_nsamples = 200
    t_begin = t_end / t_nsamples
    t = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    y = torch.sin(t)
    trajectories = y.view(1, -1, 1).repeat(trajectories_to_sample, 1, 1)
    return trajectories, t


def dde_ramp_loading_time_sol(trajectories_to_sample, device):
    t_end = 20.0
    t_nsamples = 200
    t_begin = t_end / t_nsamples
    ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    result = []
    for t in ti:
        if t < 5:
            result.append(0)
        elif 5 <= t < 10:
            result.append((1.0 / 4.0) * ((t - 5) - 0.5 * torch.sin(2 * (t - 5))))
        elif 10 <= t:
            result.append(
                (1.0 / 4.0) * ((t - 5) - (t - 10) - 0.5 * torch.sin(2 * (t - 5)) + 0.5 * torch.sin(2 * (t - 10)))
            )
    y = torch.Tensor(result).to(device).double() / 5.0
    trajectories = y.view(1, -1, 1).repeat(trajectories_to_sample, 1, 1)
    return trajectories, ti


def parse_datasets(args, device):
    def basic_collate_fn(batch, time_steps, args=args, device=device, data_type="train"):
        batch = torch.stack(batch)
        data_dict = {"data": batch, "time_steps": time_steps}

        data_dict = split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap
    max_t_extrap = args.max_t / args.timepoints * n_total_tp
    if dataset_name == "sine" or dataset_name == "dde_ramp_loading_time_sol":
        trajectories_to_sample = 1000
        if dataset_name == "sine":
            trajectories, t = sine(trajectories_to_sample, device)
        elif dataset_name == "dde_ramp_loading_time_sol":
            trajectories, t = dde_ramp_loading_time_sol(trajectories_to_sample, device)

        # # Normalise
        # samples = trajectories.shape[0]
        # dim = trajectories.shape[2]
        # traj = (trajectories.view(-1, dim) - trajectories.view(-1,
        #         dim).mean(0)) / trajectories.view(-1, dim).std(0)
        # trajectories = torch.reshape(traj, (samples, -1, dim))

        traj_index = torch.randperm(trajectories.shape[0])  # pyright: ignore
        train_split = int(0.8 * trajectories.shape[0])  # pyright: ignore
        test_split = int(0.9 * trajectories.shape[0])  # pyright: ignore
        train_trajectories = trajectories[traj_index[:train_split], :, :]  # pyright: ignore
        test_trajectories = trajectories[traj_index[test_split:], :, :]  # pyright: ignore

        # test_plot_traj = test_trajectories[0, :, :]

        input_dim = train_trajectories.shape[2]
        # output_dim = input_dim
        batch_size = 128

        train_dataloader = DataLoader(
            train_trajectories,  # pyright: ignore
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(batch, t, data_type="train"),
        )
        test_dataloader = DataLoader(
            test_trajectories,  # pyright: ignore
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(batch, t, data_type="test"),
        )

        data_objects = {
            "dataset_obj": "",
            "train_dataloader": inf_generator(train_dataloader),
            "test_dataloader": inf_generator(test_dataloader),
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(test_dataloader),
        }
        return data_objects

    ########### 1d datasets ###########

    # Sampling args.timepoints time points in the interval [0, args.max_t]
    # Sample points for both training sequence and explapolation (test)
    distribution = uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([max_t_extrap]))
    time_steps_extrap = distribution.sample(torch.Size([n_total_tp - 1]))[:, 0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]

    dataset_obj = None

    if dataset_obj is None:
        raise Exception(f"Unknown dataset: {dataset_name}")  # pylint: disable=broad-exception-raised

    dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples=args.n, noise_weight=args.noise_weight)

    # Process small datasets
    dataset = dataset.to(device).double()
    time_steps_extrap = time_steps_extrap.to(device).double()

    train_y, test_y = split_train_test(dataset, train_fraq=0.8)

    # n_samples = len(dataset)
    input_dim = dataset.size(-1)

    batch_size = min(args.batch_size, args.n)
    train_dataloader = DataLoader(
        train_y,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type="train"),
    )
    test_dataloader = DataLoader(
        test_y,
        batch_size=args.n,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type="test"),
    )

    data_objects = {  # "dataset_obj": dataset_obj,
        "train_dataloader": inf_generator(train_dataloader),
        "test_dataloader": inf_generator(test_dataloader),
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader),
    }

    return data_objects
