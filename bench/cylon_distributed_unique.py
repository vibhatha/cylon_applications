##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

import argparse
import os
import time

import numpy as np
import pandas as pd
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon.net import MPIConfig

"""
Run benchmark:

>>> mpirun -n 4 python cylon_distributed_unique.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 2_000_000 \
                                        --num_cols 2 \
                                        --stats_file dist_unique_bench_test_v2.csv \
                                        --repetitions 1 \
                                        --base_file_path ~/data/cylon_bench
                                        
"""


def unique_op(ctx, num_rows, base_file_path):
    parallelism = ctx.get_world_size()

    csv_read_options = CSVReadOptions() \
        .use_threads(True) \
        .block_size(1 << 30)

    sub_path = "records_{}/parallelism_{}".format(num_rows, parallelism)
    distributed_file_prefix = "distributed_data_file_rank_{}.csv".format(ctx.get_rank())

    file_path = os.path.join(base_file_path, sub_path, distributed_file_prefix)

    tb = read_csv(ctx, file_path, csv_read_options)

    cylon_time = time.time()
    tb2 = tb.distributed_unique()
    cylon_time = time.time() - cylon_time

    return cylon_time


def bench_unique_op(ctx, start, end, step, num_cols, repetitions, stats_file, base_file_path):
    all_data = []
    schema = ["num_records", "num_cols", "time(s)"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            cylon_time = unique_op(ctx=ctx, num_rows=records, base_file_path=base_file_path)
            times.append([cylon_time])
        times = np.array(times).sum(axis=0) / repetitions
        if ctx.get_rank() == 0:
            print("Unique Op : Records={}, Columns={}, Cylon Time : {}".format(records, num_cols, times[0]))
            all_data.append(
                [records, num_cols, times[0]])
    if ctx.get_rank() == 0:
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
    parser.add_argument("-d", "--duplication_factor",
                        help="random data duplication factor",
                        type=float)
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

    args = parser.parse_args()
    mpi_config = MPIConfig()
    ctx = CylonContext(config=mpi_config, distributed=True)
    if ctx.get_rank() == 0:
        print("Start Data Size : {}".format(args.start_size))
        print("End Data Size : {}".format(args.end_size))
        print("Step Data Size : {}".format(args.step_size))
        print("Number of Columns : {}".format(args.num_cols))
        print("Number of Repetitions : {}".format(args.repetitions))
        print("Stats File : {}".format(args.stats_file))
        print("Base File Path : {}".format(args.base_file_path))

    bench_unique_op(ctx=ctx,
                    start=args.start_size,
                    end=args.end_size,
                    step=args.step_size,
                    num_cols=args.num_cols,
                    repetitions=args.repetitions,
                    stats_file=args.stats_file,
                    base_file_path=args.base_file_path)
    ctx.finalize()
