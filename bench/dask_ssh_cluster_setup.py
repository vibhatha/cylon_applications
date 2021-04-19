import dask
import dask.dataframe as dd
from dask.distributed import Client, SSHCluster
import time

cluster = SSHCluster(["v-001", "v-002"], worker_options={"nthreads": 1, "nprocs": 16})
client = Client(cluster)

time.sleep(100)

client.close()
cluster.shutdown()
