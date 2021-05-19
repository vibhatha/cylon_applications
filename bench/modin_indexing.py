import time
import os
os.environ["MODIN_CPUS"] = "1"
os.environ['MODIN_ENGINE'] = 'ray'
import modin.pandas as pd
import numpy as np
import argparse
from bench_util import get_dataframe

"""
Run benchmark:

>>> python modin_indexing.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 10_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/modin_indexing_bench.csv \
                                        --unique_factor 0.1 \
                                        --repetitions 1
"""


def indexing_op(num_rows: int, num_cols: int, unique_factor: float):
    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, unique_factor=unique_factor)
    pdf = pd.DataFrame(pdf)
    filter_column = pdf.columns[0]
    filter_column_data = pdf[pdf.columns[0]]
    random_index = np.random.randint(low=0, high=pdf.shape[0])
    filter_value = filter_column_data.values[random_index]
    filter_values = filter_column_data.values.tolist()[0:pdf.shape[0] // 2]
    filter_values = np.unique(np.array(filter_values)).tolist()
    pdf_indexing_time = time.time()
    pdf.set_index(filter_column, drop=True, inplace=True)
    pdf_indexing_time = time.time() - pdf_indexing_time

    modin_filter_time = time.time()
    pdf_filtered = pdf.loc[filter_values]
    modin_filter_time = time.time() - modin_filter_time

    return modin_filter_time, pdf_indexing_time


def bench_indexing_op(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                      unique_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "modin_loc", "modin_indexing"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            pandas_filter_time, pdf_indexing_time = indexing_op(
                num_rows=records, num_cols=num_cols,
                unique_factor=unique_factor)
            times.append([pandas_filter_time, pdf_indexing_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(
            f"Loc Op : Records={records}, Columns={num_cols}, Modin Loc Time : {times[0]}, "
            f"Modin Indexing Time : {times[1]}")
        all_data.append(
            [records, num_cols, times[0], times[1]])
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
    bench_indexing_op(start=args.start_size,
                      end=args.end_size,
                      step=args.step_size,
                      num_cols=args.num_cols,
                      repetitions=args.repetitions,
                      stats_file=args.stats_file,
                      unique_factor=args.unique_factor)
