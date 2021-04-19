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


class DaskCluster(object):

    def __init__(self, scheduler_host, ips, memory_limit, network_interface, nprocs, nthreads, local_directory,
                 scheduler_file,
                 python_env, num_nodes, wait):
        self.scheduler_host = scheduler_host
        self.ips = ips
        self.memory_limit = memory_limit
        self.network_interface = network_interface
        self.nprocs = nprocs
        self.nthreads = nthreads
        self.local_directory = local_directory
        self.scheduler_file = scheduler_file
        self.python_env = python_env
        self.num_nodes = num_nodes
        self.wait = wait

    def start_scheduler(self):
        print("Starting Scheduler")
        subprocess.Popen(
            ["ssh", self.scheduler_host, self.python_env + "/bin/dask-scheduler", "--scheduler-file", "--interface",
             self.network_interface,
             self.scheduler_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(self.wait)

    def start_workers(self):
        for ip in self.ips[0:self.num_nodes]:
            print("Starting Worker : {}".format(ip))
            subprocess.Popen(
                ["ssh", ip, self.python_env + "/bin/dask-worker", self.scheduler_host + ":8786", "--nthreads",
                 str(self.nthreads), "--nprocs",
                 str(self.nprocs), "--memory-limit", self.memory_limit, "--interface", self.network_interface,
                 "--local-directory", self.local_directory,
                 "--scheduler-file",
                 self.scheduler_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        time.sleep(self.wait)

    def stop_scheduler(self):
        print("Stopping scheduler")
        subprocess.run(["pkill", "-f", "dask-scheduler"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(self.wait)

    def stop_workers(self):
        print("Stopping workers")
        for ip in self.ips:
            print("stopping worker", ip, flush=True)
            subprocess.run(["ssh", ip, "pkill", "-f", "dask-worker"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(self.wait)

    def start_cluster(self):
        print("Start Cluster")
        self.start_scheduler()
        print("Scheduler Started")
        self.start_workers()
        print("Workers Started")

    def stop_cluster(self):
        print("Stop Cluster")
        self.stop_workers()
        print("Stopped Workers")
        self.stop_scheduler()
        print("Stopped Scheduler")
