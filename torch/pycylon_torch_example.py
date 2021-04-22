"""
Install: PyCylon (Follow: https://cylondata.org/docs/)
Run Program: mpirun -n 4 python cylon_pytorch_demo_distributed.py
"""
import os
import argparse
import socket
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pycylon.net import MPIConfig
from pycylon import CylonEnv
from pycylon import DataFrame
from pycylon import read_csv
from torch.nn.parallel import DistributedDataParallel as DDP
from pycylon.util.logging import log_level, disable_logging

log_level(0)  # set an arbitrary log level
disable_logging()  # disable logging completely

hostname = socket.gethostname()


def setup(rank, world_size, backend, master_address, port):
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = port
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # initialize the process group
    dist.init_process_group(backend=backend, init_method="env://")
    mpi_config = MPIConfig()
    env = CylonEnv(config=mpi_config, distributed=True)
    print(f"Init Process Groups : => [{hostname}]Demo DDP Rank {rank}")
    return env


def cleanup():
    dist.destroy_process_group()


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(4, 1)
        self.hidden2 = nn.Linear(1, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


def demo_basic(rank, world_size, backend, epochs, master_address, port):
    print(f"Simple Batch Train => [{hostname}]Demo DDP Rank {rank}")
    env = setup(rank=rank, world_size=world_size, backend=backend, master_address=master_address, port=port)
    cuda_available = torch.cuda.is_available()
    device = 'cuda:' + str(rank) if cuda_available else 'cpu'
    base_path = "/tmp"

    user_devices_file = os.path.join(base_path, f'user_device_tm_{rank + 1}.csv')
    user_usage_file = os.path.join(base_path, f'user_usage_tm_{rank + 1}.csv')

    user_devices_data = read_csv(user_devices_file, sep=',')
    user_usage_data = read_csv(user_usage_file, sep=',')

    print(f"User Devices Data Rows:{len(user_devices_data)}, Columns: {len(user_devices_data.columns)}")
    print(f"User Usage Data Rows:{len(user_usage_data)}, Columns: {len(user_usage_data.columns)}")

    print("--------------------------------")
    print("Before Join")
    print("--------------------------------")
    print(user_devices_data[0:5])
    print("-------------------------------------")
    print(user_usage_data[0:5])

    join_df = user_devices_data.merge(right=user_usage_data, left_on=[0], right_on=[3], algorithm='sort')
    print("----------------------")
    print("New Table After Join (5 Records)")
    print(join_df[0:5])
    print("----------------------")

    data_ar: np.ndarray = join_df.to_numpy()

    data_features: np.ndarray = data_ar[:, 2:6]
    data_learner: np.ndarray = data_ar[:, 6:7]

    x_train, y_train = data_features[0:100], data_learner[0:100]
    x_test, y_test = data_features[100:], data_learner[100:]

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    # create model and move it to GPU with id rank

    model = Network().to(device)

    ddp_model = DDP(model, device_ids=[device]) if cuda_available else DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    if rank == 0:
        print("Training A Dummy Model")
    for t in range(epochs):
        for x_batch, y_batch in zip(x_train, y_train):
            print(f"Epoch {t}", end='\r')
            prediction = ddp_model(x_batch)
            loss = loss_fn(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend",
                        help="example : 'mpi', 'nccl'",
                        default='mpi',
                        type=str)
    parser.add_argument("-e", "--epochs",
                        help="training epochs",
                        default=10,
                        type=int)
    parser.add_argument("-m", "--master_address",
                        help="master address for torch distributed runtime",
                        default='localhost',
                        type=str)
    parser.add_argument("-p", "--port",
                        help="torch port for distributed runtime",
                        default='12335',
                        type=str)
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    backend = args.backend
    demo_basic(rank=rank, world_size=world_size, backend=backend, epochs=args.epochs,
                     master_address=args.master_address, port=args.port)

    cleanup()
