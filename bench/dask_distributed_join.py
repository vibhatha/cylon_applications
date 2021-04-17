import os

import dask
import dask.dataframe as dd
from dask.distributed import Client, SSHCluster
import pandas as pd
import time
import argparse
import math
import subprocess
import numpy as np

"""
>>> python dask_setup.py --start_size 100_000_000 \
                                        --step_size 100_000_000 \
                                        --end_size 500_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/dask_dist_join_bench.csv \
                                        --repetitions 3 \
                                        --base_file_path ~/data/cylon_bench \
                                        --parallelism 64 \
                                        --nodes_file /tmp/hostfile \
                                        --total_nodes 8 \
                                        --scheduler_host v-001 \
                                        --python_env /home/vibhatha/venv/ENVCYLON
"""


def get_ips(nodes_file):
    ips = []
    with open(nodes_file, 'r') as fp:
        for l in fp.readlines():
            ips.append(l.split(' ')[0])
    return ips


def start_cluster(ips, scheduler_host, python_env, procs, nodes):
    print("starting scheduler", flush=True)
    # subprocess.Popen(
    #     ["ssh", "v-001", "/N/u2/d/dnperera/victor/git/cylon/ENV/bin/dask-scheduler", "--interface", "enp175s0f0",
    #      "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    subprocess.Popen(
        ["ssh", scheduler_host, python_env + "/bin/dask-scheduler"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    for ip in ips[0:nodes]:
        print("starting worker", ip, flush=True)
        # subprocess.Popen(
        #     ["ssh", ip, "/N/u2/d/dnperera/victor/git/cylon/ENV/bin/dask-worker", "v-001:8786", "--interface",
        #      "enp175s0f0", "--nthreads", "1", "--nprocs", str(procs), "--memory-limit", "20GB", "--local-directory",
        #      "/scratch/dnperera/dask/", "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT)
        subprocess.Popen(
            ["ssh", ip, python_env + "/bin/dask-worker", scheduler_host + ":8786", "--nthreads", "1", "--nprocs",
             str(procs), "--memory-limit", "20GB", "--local-directory", "/scratch/vlabeyko/dask/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    time.sleep(5)


def stop_cluster(ips):
    for ip in ips:
        print("stopping worker", ip, flush=True)
        subprocess.run(["ssh", ip, "pkill", "-f", "dask-worker"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    print("stopping scheduler", flush=True)
    subprocess.run(["pkill", "-f", "dask-scheduler"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(5)


def dask_join(scheduler_host, num_rows, base_file_path, num_nodes):

    def join_func():
        sub_path = "records_{}/parallelism_{}".format(num_rows, parallelism)
        distributed_file_prefix = "single_data_file.csv"
        left_file_path = os.path.join(base_file_path, sub_path, distributed_file_prefix)
        right_file_path = os.path.join(base_file_path, sub_path, distributed_file_prefix)
        df_l = dd.read_csv(left_file_path).repartition(npartitions=num_nodes)
        df_r = dd.read_csv(right_file_path).repartition(npartitions=num_nodes)

        client.persist([df_l, df_r])

        print("left rows", len(df_l), flush=True)
        print("right rows", len(df_r), flush=True)
        join_time = time.time()
        out = df_l.merge(df_r, on='0', how='inner', suffixes=('_left', '_right')).compute()
        join_time = time.time() - join_time
        return join_time

    client = Client(scheduler_host + ':8786')
    print(client)
    dask_time_future = client.submit(join_func)
    dask_time = dask_time_future.result()
    client.close()
    return dask_time


def bench_join_op(start, end, step, num_cols, repetitions, stats_file, base_file_path, num_nodes):
    all_data = []
    schema = ["num_records", "num_cols", "time(s)"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            dask_time = dask_join(scheduler_host=scheduler_host, num_rows=records, base_file_path=base_file_path,
                                  num_nodes=num_nodes)
            times.append([dask_time])
        times = np.array(times).sum(axis=0) / repetitions
        print("Join Op : Records={}, Columns={}, Dask Time : {}".format(records, num_cols, times[0]))
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
    parser.add_argument("-nf", "--nodes_file",
                        help="nodes file",
                        type=str)
    parser.add_argument("-sh", "--scheduler_host",
                        help="scheduler host",
                        type=str)
    parser.add_argument("-pe", "--python_env",
                        help="python env",
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
    print("Parallelism : {}".format(args.parallelism))
    print("Nodes File : {}".format(args.nodes_file))
    print("Scheduler Host : {}".format(args.scheduler_host))
    print("Python ENV : {}".format(args.python_env))

    parallelism = args.parallelism
    TOTAL_NODES = args.total_nodes
    procs = int(math.ceil(parallelism / TOTAL_NODES))
    nodes = min(parallelism, TOTAL_NODES)
    ips = get_ips(args.nodes_file)
    python_env = args.python_env
    scheduler_host = args.scheduler_host
    print("NODES : ", ips)
    print("Processes Per Node: ", procs)

    stop_cluster(ips)
    start_cluster(ips=ips, scheduler_host=scheduler_host, python_env=python_env, procs=procs, nodes=nodes)
    bench_join_op(start=args.start_size,
                  end=args.end_size,
                  step=args.step_size,
                  num_cols=args.num_cols,
                  repetitions=args.repetitions,
                  stats_file=args.stats_file,
                  base_file_path=args.base_file_path,
                  num_nodes=args.total_nodes)
    stop_cluster(ips)
