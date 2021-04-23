"""
Install: PyCylon (Follow: https://cylondata.org/docs/)
Run Program:  horovodrun -np 4 python3 pycylon_tensorflow_example.py --epochs 20
References:
    1. https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_mnist.py
    2. https://horovod.readthedocs.io/en/stable/tensorflow.html
"""
import argparse
import os
import socket

import numpy as np
import pandas as pd
from pycylon import CylonEnv
from pycylon import DataFrame
from pycylon.net import MPIConfig
from pycylon.util.logging import log_level, disable_logging
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import horovod.tensorflow as hvd

log_level(0)  # set an arbitrary log level
disable_logging()  # disable logging completely

hostname = socket.gethostname()


def setup():
    hvd.init()
    assert hvd.mpi_threads_supported()
    mpi_config = MPIConfig()
    env = CylonEnv(config=mpi_config, distributed=True)
    rank = env.rank
    world_size = env.world_size
    print(f"Init Process Groups : => [{hostname}]Demo DDP Rank: {rank} , World Size: {world_size}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    return env


def demo_basic(epochs):
    env = setup()
    rank = env.rank
    print(f"Simple Batch Train => [{hostname}]Demo DDP Rank {rank}")

    # device = 'cuda:' + str(rank) if cuda_available else 'cpu'
    base_path = "https://raw.githubusercontent.com/cylondata/cylon/main/cpp/src/tutorial/data/"

    user_devices_file = os.path.join(base_path, f'user_device_tm_{rank + 1}.csv')
    user_usage_file = os.path.join(base_path, f'user_usage_tm_{rank + 1}.csv')
    print("Rank[{}] User Device File : {}".format(rank, user_devices_file))
    print("Rank[{}] User Usage File : {}".format(rank, user_usage_file))
    user_devices_data = DataFrame(pd.read_csv(user_devices_file))  # read_csv(user_devices_file, sep=',')
    user_usage_data = DataFrame(pd.read_csv(user_usage_file))  # read_csv(user_usage_file, sep=',')

    print(f"Rank [{rank}] User Devices Data Rows:{len(user_devices_data)}, Columns: {len(user_devices_data.columns)}")
    print(f"Rank [{rank}] User Usage Data Rows:{len(user_usage_data)}, Columns: {len(user_usage_data.columns)}")

    print("--------------------------------")
    print("Before Join")
    print("--------------------------------")
    print(user_devices_data[0:5])
    print("-------------------------------------")
    print(user_usage_data[0:5])

    join_df = user_devices_data.merge(right=user_usage_data, left_on=[0], right_on=[3], algorithm='hash')
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

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    print("=" * 80)
    print("Tensorflow DataSets")
    print("=" * 80)

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # define network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3), tf.keras.layers.Dense(1)])
    # define loss function
    loss = tf.losses.MeanSquaredError()
    # define optimizer
    opt = tf.optimizers.Adam(0.001 * hvd.size())

    @tf.function
    def training_step(images, labels, first_batch):
        # define a step function for training
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss(labels, probs)

        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value

    if rank == 0:
        print("Training A Dummy Model")
    take_count = x_train.shape[0] // hvd.size()
    for t in range(epochs):
        for batch, (images, labels) in enumerate(train_dataset.take(take_count)):
            loss_value = training_step(images, labels, batch == 0)
            if batch % 10 == 0 and hvd.local_rank() == 0:
                print("Epoch : {}, Batch : {}, Loss : {}".format(t, batch, loss_value))

    if rank == 0:
        print("Data Analysis Complete!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs",
                        help="training epochs",
                        default=10,
                        type=int)
    args = parser.parse_args()
    demo_basic(epochs=args.epochs)
