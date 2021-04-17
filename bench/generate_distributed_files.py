import argparse
import os

import numpy as np

from bench_util import get_dataframe
from bench_util import line_separator

"""
python3 generate_distributed_files.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 4_000_000 \
                                        --unique_factor 0.1 \
                                        --num_cols 2 \
                                        --file_path ~/data/cylon_bench \
                                        --parallelism 4
"""


def generation_op(start: int, end: int, step: int, num_cols: int, file_path: str,
                  unique_factor: float, parallelism: int):
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        line_separator()
        print("Generating Records : {}".format(records))
        line_separator()
        sequential_file = "single_data_file.csv"
        distributed_file_prefix = "distributed_data_file"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        record_file_path = os.path.join(file_path, "records_{}".format(str(records)))
        if not os.path.exists(record_file_path):
            os.mkdir(record_file_path)
        new_sub_data_dir_path = os.path.join(record_file_path, "parallelism_{}".format(parallelism))
        if not os.path.exists(new_sub_data_dir_path):
            os.mkdir(new_sub_data_dir_path)
        pdf = get_dataframe(num_rows=records, num_cols=num_cols, unique_factor=unique_factor)
        pdf_splits = np.array_split(pdf, parallelism)
        for rank in range(parallelism):
            distributed_file_name = distributed_file_prefix + "_rank_{}.csv".format(rank)
            dist_file_save_path = os.path.join(new_sub_data_dir_path, distributed_file_name)
            seq_file_save_path = os.path.join(new_sub_data_dir_path, sequential_file)
            print(pdf.shape, pdf_splits[rank].shape)
            pdf.to_csv(seq_file_save_path, sep=",", index=False)
            pdf_splits[rank].to_csv(dist_file_save_path, sep=",", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_size",
                        help="initial data size",
                        type=int)
    parser.add_argument("-e", "--end_size",
                        help="end data size",
                        type=int)
    parser.add_argument("-d", "--unique_factor",
                        help="random data uniqueness factor",
                        type=float)
    parser.add_argument("-s", "--step_size",
                        help="Step size",
                        type=int)
    parser.add_argument("-c", "--num_cols",
                        help="number of columns",
                        type=int)
    parser.add_argument("-fp", "--file_path",
                        help="file path",
                        type=str)
    parser.add_argument("-p", "--parallelism",
                        help="number of processes",
                        type=int)

    args = parser.parse_args()
    print(f"Start Data Size : {args.start_size}")
    print(f"End Data Size : {args.end_size}")
    print(f"Step Data Size : {args.step_size}")
    print(f"Data Duplication Factor : {args.unique_factor}")
    print(f"Number of Columns : {args.num_cols}")
    print(f"File Path : {args.file_path}")
    print(f"Parallelism : {args.parallelism}")
    generation_op(start=args.start_size,
                  end=args.end_size,
                  step=args.step_size,
                  num_cols=args.num_cols,
                  file_path=args.file_path,
                  parallelism=args.parallelism,
                  unique_factor=args.unique_factor
                  )
