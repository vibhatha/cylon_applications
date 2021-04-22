"""
Install: PyCylon (Follow: https://cylondata.org/docs/)
Run Program: mpirun -n 4 python cylon_pytorch_demo_distributed.py
"""
import argparse
import os
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pycylon import CylonEnv
from pycylon import read_csv
from pycylon.net import MPIConfig
from pycylon.util.logging import log_level, disable_logging
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DistributedDataParallel as DDP

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
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        y_pred = F.leaky_relu(self.linear(x))
        return y_pred


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

    print(f"Rank [{rank}] User Devices Data Rows:{len(user_devices_data)}, Columns: {len(user_devices_data.columns)}")
    print(f"Rank [{rank}] User Usage Data Rows:{len(user_usage_data)}, Columns: {len(user_usage_data.columns)}")

    print("--------------------------------")
    print("Before Join")
    print("--------------------------------")
    print(user_devices_data[0:5])
    print("-------------------------------------")
    print(user_usage_data[0:5])

    join_df = user_devices_data.merge(right=user_usage_data, left_on=[0], right_on=[3], algorithm='sort')
    print("----------------------")
    print("Rank [{}] New Table After Join (5 Records)".format(rank))
    print(join_df[0:5])
    print("----------------------")
    feature_df = join_df[
        ['_xplatform_version', '_youtgoing_mins_per_month', '_youtgoing_sms_per_month',
         '_ymonthly_mb']]
    feature_df.rename(
        ['platform_version', 'outgoing_mins_per_month', 'outgoing_sms_per_month', 'monthly_mb'])
    if rank == 0:
        print("Data Engineering Complete!!!")
    print("=" * 80)
    print("Rank [{}] Feature DataFrame ".format(rank))
    print(feature_df[0:5])
    print("=" * 80)
    data_ar: np.ndarray = feature_df.to_numpy()

    data_features: np.ndarray = data_ar[:, 0:3]
    data_learner: np.ndarray = data_ar[:, 3:4]

    x_train, y_train = data_features[0:100], data_learner[0:100]
    x_test, y_test = data_features[100:], data_learner[100:]

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    sc = StandardScaler()
    sct = StandardScaler()
    x_train = sc.fit_transform(x_train)
    y_train = sct.fit_transform(y_train)
    x_test = sc.fit_transform(x_test)
    y_test = sct.fit_transform(y_test)

    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    # create model and move it to GPU with id rank

    model = Network().to(device)
    ddp_model = DDP(model, device_ids=[device]) if cuda_available else DDP(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    optimizer.zero_grad()
    if rank == 0:
        print("Training A Dummy Model")
    for t in range(epochs):
        for x_batch, y_batch in zip(x_train, y_train):
            print(f"Epoch {t}", end='\r')
            prediction = ddp_model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if rank == 0:
        print("Testing A Dummy Model")
    prediction_model = ddp_model.train(False)
    for x_batch, y_batch in zip(x_test, y_test):
        prediction = prediction_model(x_batch)
        inv_prediction = sct.inverse_transform(prediction.detach().numpy().reshape(-1, 1))
        inv_ybatch = sct.inverse_transform(y_batch.detach().numpy().reshape(-1, 1))
        inv_prediction = inv_prediction.reshape(1)[0]
        inv_ybatch = inv_ybatch.reshape(1)[0]

    if rank == 0:
        print("Data Analysis Complete!!!")


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
