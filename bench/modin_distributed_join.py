import os
import modin.pandas as pd
import time
import argparse
import math
import subprocess
import numpy as np
import json

"""
Ray Head: ray start --head --port 12345

Ray Worker: ray start --address='v-015:12345' --redis-password='5241590000000000' --num-cpus 32
"""

"""
>>> python modin_distributed_join.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 2_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/dask_dist_join_bench.csv \
                                        --repetitions 1 \
                                        --base_file_path ~/data/cylon_bench \
                                        --parallelism 4 \
                                        --nodes_file /hostfiles/hostfile_victor_8x16 \
                                        --total_nodes 1 \
                                        --scheduler_host v-001 \
                                        --memory_limit_per_worker 4G \
                                        --python_env /home/vibhatha/venv/ENVCYLON
"""


def get_ips(nodes_file):
    ips = []
    with open(nodes_file, 'r') as fp:
        for l in fp.readlines():
            ips.append(l.split(' ')[0])
    return ips


def modin_join(num_rows, base_file_path, parallelism):
    print("Join Function")
    sub_path = "records_{}/parallelism_{}".format(num_rows, parallelism)
    distributed_file_prefix = "single_data_file.csv"
    left_file_path = os.path.join(base_file_path, sub_path, distributed_file_prefix)
    right_file_path = os.path.join(base_file_path, sub_path, distributed_file_prefix)
    print("Reading files...")
    df_l = pd.read_csv(left_file_path)
    df_r = pd.read_csv(right_file_path)

    join_column = df_l.columns[0]

    print("left rows", len(df_l), flush=True)
    print("right rows", len(df_r), flush=True)
    join_time = time.time()
    out = df_l.merge(df_r, on=join_column, how='inner', suffixes=('_left', '_right'))
    join_time = time.time() - join_time
    return join_time


def bench_join_op(start, end, step, num_cols, repetitions, stats_file, base_file_path, parallelism):
    all_data = []
    schema = ["num_records", "num_cols", "time(s)"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            dask_time = modin_join(num_rows=records, base_file_path=base_file_path, parallelism=parallelism)
            times.append([dask_time])
        times = np.array(times).sum(axis=0) / repetitions
        print("Join Op : Records={}, Columns={}, Modin Time : {}".format(records, num_cols, times[0]))
        all_data.append([records, num_cols, times[0]])
    pdf = pd.DataFrame(all_data, columns=schema)
    print(pdf)
    pdf.to_csv(stats_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_size",
                        help="initial data size",
                        type=int)
    parser.add_argument("-e", "--end_size",
                        help="end data size",
                        type=int)
    parser.add_argument("-s", "--step_size",
                        help="Step size",
                        type=int)
    parser.add_argument("-c", "--num_cols",
                        help="number of columns",
                        type=int)
    parser.add_argument("-r", "--repetitions",
                        help="number of experiments to be repeated",
                        type=int)
    parser.add_argument("-f", "--stats_file",
                        help="stats file to be saved",
                        type=str)
    parser.add_argument("-bf", "--base_file_path",
                        help="base file path",
                        type=str)
    parser.add_argument("-p", "--parallelism",
                        help="parallelism",
                        type=int)
    parser.add_argument("-n", "--total_nodes",
                        help="total nodes",
                        type=int)
    parser.add_argument("-ml", "--memory_limit_per_worker",
                        help="memory limit per worker",
                        type=str)
    parser.add_argument("-ni", "--network_interface",
                        help="network interface",
                        type=str)
    parser.add_argument("-nf", "--nodes_file",
                        help="nodes file",
                        type=str)
    parser.add_argument("-sh", "--scheduler_host",
                        help="scheduler host",
                        type=str)
    parser.add_argument("-pe", "--python_env",
                        help="python env",
                        type=str)
    parser.add_argument("-cl", "--cluster",
                        help="cluster mode",
                        type=str)

    args = parser.parse_args()
    print("Start Data Size : {}".format(args.start_size))
    print("End Data Size : {}".format(args.end_size))
    print("Step Data Size : {}".format(args.step_size))
    print("Number of Columns : {}".format(args.num_cols))
    print("Number of Repetitions : {}".format(args.repetitions))
    print("Stats File : {}".format(args.stats_file))
    print("Base File Path : {}".format(args.base_file_path))
    print("Total Nodes : {}".format(args.total_nodes))
    print("Memory limit per worker : {}".format(args.memory_limit_per_worker))
    print("Network Interface : {}".format(args.network_interface))
    print("Parallelism : {}".format(args.parallelism))
    print("Nodes File : {}".format(args.nodes_file))
    print("Scheduler Host : {}".format(args.scheduler_host))
    print("Python ENV : {}".format(args.python_env))
    print("Cluster : {}".format(args.cluster))

    parallelism = args.parallelism
    os.environ["MODIN_CPUS"] = str(parallelism)
    TOTAL_NODES = args.total_nodes
    procs = int(math.ceil(parallelism / TOTAL_NODES))
    nodes = min(parallelism, TOTAL_NODES)
    # ips = get_ips(args.nodes_file)
    python_env = args.python_env
    scheduler_host = args.scheduler_host
    # print("NODES : ", ips)
    print("Processes Per Node: ", procs)
    # if args.cluster == 'ray':
    #     # ray.init()
    #     ray.init(address='auto', _redis_password='5241590000000000')
    # elif args.cluster == 'dask':
    #     from distributed import Client
    #     client = Client(args.scheduler_host + ':8786')
    # else:
    #     ray.init(
    #         _system_config={
    #             "object_spilling_config": json.dumps(
    #                 {"type": "filesystem", "params": {"directory_path": "/scratch/vlabeyko/modin"}},
    #             )
    #         },
    #     )

    bench_join_op(start=args.start_size,
                  end=args.end_size,
                  step=args.step_size,
                  num_cols=args.num_cols,
                  repetitions=args.repetitions,
                  stats_file=args.stats_file,
                  base_file_path=args.base_file_path,
                  parallelism=parallelism)
