from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


data = [i * (rank + 1) for i in range(3)]
all_data = comm.allgather(data)
np_all_data = np.array(all_data).flatten()
np_all_unique_data = np.unique(np_all_data)
print(all_data, np_all_unique_data)