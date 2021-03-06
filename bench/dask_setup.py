import os

import dask
import dask.dataframe as dd
from dask.distributed import Client, SSHCluster

import time
import argparse
import math
import subprocess

"""
>>> python dask_setup.py --start_size 100_000_000 \
                                        --step_size 100_000_000 \
                                        --end_size 500_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/dask_dist_join_bench.csv \
                                        --repetitions 3 \
                                        --base_file_path ~/data/cylon_bench \
                                        --parallelism 4 \
                                        --nodes_file /tmp/hostfile \
                                        --total_nodes 1 \
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
        ["ssh", scheduler_host, python_env + "/bin/dask-scheduler", "--scheduler-file",
         "/N/u2/v/vlabeyko/dask-sched.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(15)

    for ip in ips[0:nodes]:
        print("starting worker", ip, flush=True)
        # subprocess.Popen(
        #     ["ssh", ip, "/N/u2/d/dnperera/victor/git/cylon/ENV/bin/dask-worker", "v-001:8786", "--interface",
        #      "enp175s0f0", "--nthreads", "1", "--nprocs", str(procs), "--memory-limit", "20GB", "--local-directory",
        #      "/scratch/dnperera/dask/", "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT)
        subprocess.Popen(
            ["ssh", ip, python_env + "/bin/dask-worker", scheduler_host + ":8786", "--nthreads", "1", "--nprocs",
             str(procs), "--memory-limit", "20GB", "--local-directory", "/scratch/vlabeyko/dask/", "--scheduler-file", "/N/u2/v/vlabeyko/dask-sched.json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    time.sleep(15)


def stop_cluster(ips):
    for ip in ips:
        print("stopping worker", ip, flush=True)
        subprocess.run(["ssh", ip, "pkill", "-f", "dask-worker"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    print("stopping scheduler", flush=True)
    subprocess.run(["pkill", "-f", "dask-scheduler"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(5)


def dask_test_app(scheduler_host):
    def func():
        df = dask.datasets.timeseries()
        df2 = df[df.y > 0]
        df3 = df2.groupby('name').x.std()
        computed_df = df3.compute()
        return computed_df

    client = Client(scheduler_host + ':8786')
    print(client)

    future = client.submit(func)

    result = future.result()

    print(result)
    client.close()


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
    print(ips)

    stop_cluster(ips)
    start_cluster(ips=ips, scheduler_host=scheduler_host, python_env=python_env, procs=procs, nodes=nodes)
    dask_test_app(scheduler_host=scheduler_host)
    time.sleep(240)
    stop_cluster(ips)
