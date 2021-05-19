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
cpus = os.environ.get('MODIN_CPUS')
print("CPUS : {}".format(cpus))
os.environ["MODIN_CPUS"] = str(cpus)
os.environ['MODIN_ENGINE'] = 'ray'
import time
import modin.pandas as pd
import numpy as np
from bench_util import get_dataframe
import pyarrow as pa
import argparse

"""
Run benchmark:

>>> MODIN_CPUS=1 python modin_drop_duplicates.py --start_size 30_000_000 \
                                        --step_size 30_000_000 \
                                        --end_size 30_000_000 \
                                        --num_cols 2 \
                                        --filter_size 500_000 \
                                        --stats_file /tmp/modin_duplicate_bench.csv \
                                        --repetitions 5 \
                                        --unique_factor 0.1
"""


def duplicate_op(num_rows: int, num_cols: int, filter_size: int, unique_factor: float):

    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, unique_factor=unique_factor)
    pdf = pd.DataFrame(pdf)
    filter_columns = pdf.columns[0:filter_size]

    modin_time = time.time()
    pdf2 = pdf.drop_duplicates(subset=filter_columns)
    modin_time = time.time() - modin_time

    return modin_time


def bench_duplicate_op(start: int, end: int, step: int, num_cols: int, filter_size: int, repetitions: int,
                       stats_file: str,
                       unique_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "filter_size", "modin"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    assert filter_size > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            modin_time = duplicate_op(num_rows=records, num_cols=num_cols,
                                                                     filter_size=filter_size,
                                                                     unique_factor=unique_factor)
            times.append([modin_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"Duplicate Op : Records={records}, Columns={num_cols}, Filter Size={filter_size}, "
              f"Modin Time : {times[0]}")
        all_data.append(
            [records, num_cols, filter_size, times[0]])
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
    parser.add_argument("-t", "--filter_size",
                        help="number of values per filter",
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
    bench_duplicate_op(start=args.start_size,
                       end=args.end_size,
                       step=args.step_size,
                       num_cols=args.num_cols,
                       filter_size=args.filter_size,
                       repetitions=args.repetitions,
                       stats_file=args.stats_file,
                       unique_factor=args.unique_factor)
