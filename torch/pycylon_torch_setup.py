import os
from pycylon.net import MPIConfig
from pycylon import CylonEnv
import torch.distributed as dist
import socket
from pycylon.util.logging import log_level, disable_logging

log_level(0)  # set an arbitrary log level
# disable_logging()  # disable logging completely


master_address = 'localhost'
port = '12355'
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
backend = 'mpi'
hostname = socket.gethostname()
#
os.environ['MASTER_ADDR'] = master_address
os.environ['MASTER_PORT'] = port
os.environ["LOCAL_RANK"] = str(rank)
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
# initialize the process group
dist.init_process_group(backend=backend, init_method="env://")
mpi_config = MPIConfig()
env = CylonEnv(config=mpi_config, distributed=True)
print(f"Init Process Groups : => [{hostname}]Demo DDP Rank {env.rank}")

dist.destroy_process_group()
# NOTE: calling env.finalize() with our without destroy_process_group an exception occurs.

"""
Init Process Groups : => [vibhatha]Demo DDP Rank 0
Init Process Groups : => [vibhatha]Demo DDP Rank 2
Init Process Groups : => [vibhatha]Demo DDP Rank 3
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0422 12:19:49.765537 2686345 mpi_communicator.cpp:62] calling mpi finalize...
I0422 12:19:49.765576 2686345 mpi_communicator.cpp:66] Is not Finalized
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0422 12:19:49.765537 2686347 mpi_communicator.cpp:62] calling mpi finalize...
I0422 12:19:49.765576 2686347 mpi_communicator.cpp:66] Is not Finalized
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0422 12:19:49.765537 2686348 mpi_communicator.cpp:62] calling mpi finalize...
I0422 12:19:49.765576 2686348 mpi_communicator.cpp:66] Is not Finalized
Init Process Groups : => [vibhatha]Demo DDP Rank 1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0422 12:19:49.765707 2686346 mpi_communicator.cpp:62] calling mpi finalize...
I0422 12:19:49.765741 2686346 mpi_communicator.cpp:66] Is not Finalized
*** The MPI_Finalize() function was called after MPI_FINALIZE was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
[vibhatha:2686347] Local abort after MPI_FINALIZE started completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
*** The MPI_Finalize() function was called after MPI_FINALIZE was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
[vibhatha:2686348] Local abort after MPI_FINALIZE started completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
*** The MPI_Finalize() function was called after MPI_FINALIZE was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
[vibhatha:2686345] Local abort after MPI_FINALIZE started completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
*** The MPI_Finalize() function was called after MPI_FINALIZE was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
[vibhatha:2686346] Local abort after MPI_FINALIZE started completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
--------------------------------------------------------------------------
mpirun detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[22319,1],2]
  Exit code:    1
"""

