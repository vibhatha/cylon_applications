# Install

1. [Install PyCylon](https://cylondata.org/docs/conda)
2. Install Horovod with PyTorch, Tensorflow and MPI

For CPUs 

```bash
 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 pip install horovod[tensorflow,pytorch]
```

For GPUs

```bash
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 pip install horovod[tensorflow,pytorch]
```