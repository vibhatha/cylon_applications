import time
import pandas as pd
import numpy as np
from bench_util import get_dataframe
import pyarrow as pa
import argparse

import cupy as cp
import pandas as pd
import cudf
import dask_cudf

cp.random.seed(12)

"""
Run benchmark:
>>> python cudf_join.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 10_000_000 \
                                        --num_cols 2 \
                                        --stats_file cudf_join_bench.csv \
                                        --repetitions 1 \
                                        --unique_factor 0.1
"""


def join_op(num_rows: int, num_cols: int, unique_factor: float):
    pdf_left = get_dataframe(num_rows=num_rows, num_cols=num_cols, unique_factor=unique_factor,
                             stringify=False)
    pdf_right = get_dataframe(num_rows=num_rows, num_cols=num_cols, unique_factor=unique_factor,
                              stringify=False)
    # NOTE: sort join breaks when loaded data in-memory via Pandas dataframe
    gdf_left = cudf.DataFrame.from_pandas(pdf_left)
    gdf_right = cudf.DataFrame.from_pandas(pdf_right)
    join_col = pdf_left.columns[0]

    cudf_time = time.time()
    merged = gdf_left.merge(gdf_right, on=[join_col], how='inner')
    cudf_time = time.time() - cudf_time
    return cudf_time


def bench_join_op(start: int, end: int, step: int, num_cols: int, repetitions: int,
                  stats_file: str,
                  unique_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "cudf"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            cudf_time = join_op(num_rows=records, num_cols=num_cols,
                                unique_factor=unique_factor)
            times.append([cudf_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"Join Op : Records={records}, Columns={num_cols}, "
              f"Cudf Time : {times[0]}")
        all_data.append(
            [records, num_cols, times[0]])
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
                        default=0.1,
                        help="random data duplication factor",
                        type=float)
    parser.add_argument("-s", "--step_size",
                        help="Step size",
                        type=int)
    parser.add_argument("-c", "--num_cols",
                        default=2,
                        help="number of columns",
                        type=int)
    parser.add_argument("-r", "--repetitions",
                        default=1,
                        help="number of experiments to be repeated",
                        type=int)
    parser.add_argument("-f", "--stats_file",
                        default="cudf_join_stats.csv",
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
    bench_join_op(start=args.start_size,
                  end=args.end_size,
                  step=args.step_size,
                  num_cols=args.num_cols,
                  repetitions=args.repetitions,
                  stats_file=args.stats_file,
                  unique_factor=args.unique_factor)
