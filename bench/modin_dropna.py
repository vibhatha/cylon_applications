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

import os
os.environ["MODIN_CPUS"] = "1"
os.environ['MODIN_ENGINE'] = 'ray'
import pyarrow as pa
import numpy as np
import modin.pandas as pd
from bench_util import get_dataframe
import time
import argparse

"""
Run benchmark:

>>> python python/examples/op_benchmark/dropna_benchmark.py --start_size 10_000_000 \
                                        --step_size 10_000_000 \
                                        --end_size 50_000_000 \
                                        --num_cols 3 \
                                        --stats_file /tmp/dropna_bench.csv \
                                        --repetitions 1 \
                                        --unique_factor 1.0
"""


def null_handling_op(num_rows: int, num_cols: int, unique_factor: float):

    df = get_dataframe(num_rows=num_rows, num_cols=num_cols, unique_factor=unique_factor, with_null=True)
    index_df = get_dataframe(num_rows=num_rows, num_cols=1, unique_factor=1.0, with_null=False)
    index_column = 'index_col'
    df[index_column] = index_df

    df.set_index(index_column, inplace=True, drop=True)

    pandas_time = time.time()
    df_isna = df.dropna(axis=1)
    pandas_time = time.time() - pandas_time

    return pandas_time


def bench_null_handling(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                        unique_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "modin"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0

    for records in range(start, end + step, step):
        print(f"DropNa Op : Records={records}, Columns={num_cols}")
        times = []
        for idx in range(repetitions):
            pandas_time = null_handling_op(num_rows=records, num_cols=num_cols,
                                                                         unique_factor=unique_factor)
            times.append([pandas_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"DropNa Op : Records={records}, Columns={num_cols}, "
              f"Modin Time : {times[0]} s")
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
    parser.add_argument("-d", "--unique_factor",
                        help="random data unique factor",
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
    args = parser.parse_args()
    print(f"Start Data Size : {args.start_size}")
    print(f"End Data Size : {args.end_size}")
    print(f"Step Data Size : {args.step_size}")
    print(f"Data Unique Factor : {args.unique_factor}")
    print(f"Number of Columns : {args.num_cols}")
    print(f"Number of Repetitions : {args.repetitions}")
    print(f"Stats File : {args.stats_file}")
    bench_null_handling(start=args.start_size,
                        end=args.end_size,
                        step=args.step_size,
                        num_cols=args.num_cols,
                        repetitions=args.repetitions,
                        stats_file=args.stats_file,
                        unique_factor=args.unique_factor)
