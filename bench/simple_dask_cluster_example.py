from dask_cluster import DaskCluster
from dask.distributed import Client
import time

scheduler_host = "v-001"
ips = ['v-001', 'v-002', 'v-003', 'v-004']
memory_limit = '15GB'
network_interface = 'enp175s0f0'
nprocs = 16
nthreads = 1
local_directory = '/scratch/vlabeyko/dask'
scheduler_file = '/N/u2/v/vlabeyko/dask-sched.json'
python_env = '~/sandbox/UNOMT/cylon_source/cylon/ENVCYLON'
num_nodes = 4
wait = 20

dask_cluster = DaskCluster(scheduler_host=scheduler_host, ips=ips, memory_limit=memory_limit,
                           network_interface=network_interface, nprocs=nprocs, nthreads=nthreads,
                           local_directory=local_directory,
                           scheduler_file=scheduler_file, python_env=python_env, num_nodes=num_nodes, wait=wait)

dask_cluster.start_cluster()
time.sleep(100)
client = Client(scheduler_host + ":8786")
print(client)
dask_cluster.stop_cluster()
