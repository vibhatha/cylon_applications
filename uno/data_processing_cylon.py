import os
import time
import download_util
from pycylon import CylonContext
from pycylon import Table
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import numpy as np
import pandas as pd

ctx: CylonContext = CylonContext(config=None, distributed=False)


def load_aggregated_single_response_cylon(target='AUC', min_r2_fit=0.3, max_ec50_se=3.0,
                                          combo_format=False,
                                          rename=True):
    url = "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_single_response_agg"
    output_combined_single_response = \
        "/home/vibhatha/data/uno/Pilot1/workload_1/combined_single_response_agg"

    if not os.path.exists(output_combined_single_response):
        download_util.download(url=url, output_file=output_combined_single_response)

    if os.path.exists(output_combined_single_response):
        print(f"Data file : {output_combined_single_response}")
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t")
        t1 = time.time()
        tb: Table = read_csv(ctx, output_combined_single_response, csv_read_options)
        t2 = time.time()
        tb = tb[(tb['R2fit'] >= min_r2_fit) & (tb['EC50se'] <= max_ec50_se)]
        t3 = time.time()
        table_read_time = t2 - t1
        filter_time = t3 - t2
        tb = tb[['SOURCE', 'CELL', 'DRUG', target, 'STUDY']]
        tb = tb[~tb[target].isnull()]
        print("Cylon ", tb.row_count, tb.column_count, tb.column_names)

        print("Cylon Data Loading Time: ", table_read_time)
        print("Cylon Data Filter Time: ", filter_time)
        if combo_format:
            tb = tb.rename(columns={'DRUG': 'DRUG1'})
            tb['DRUG2'] = np.nan
            # tb['DRUG2'] = tb['DRUG2'].astype(object)
            tb = tb[['SOURCE', 'CELL', 'DRUG1', 'DRUG2', target, 'STUDY']]
            if rename:
                tb = tb.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                        'DRUG1': 'Drug1', 'DRUG2': 'Drug2', 'STUDY': 'Study'})
        else:
            if rename:
                tb = tb.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                        'DRUG': 'Drug', 'STUDY': 'Study'})


def load_single_dose_response_cylon(combo_format=False, fraction=True):
    url = "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/rescaled_combined_single_drug_growth"
    rescaled_combined_single_drug_growth = \
        "/home/vibhatha/data/uno/Pilot1/workload_1/rescaled_combined_single_drug_growth"

    if not os.path.exists(rescaled_combined_single_drug_growth):
        download_util.download(url=url, output_file=rescaled_combined_single_drug_growth)
    if os.path.exists(rescaled_combined_single_drug_growth):
        print("------------------Cylon--------------------")
        print(f"Data file : {rescaled_combined_single_drug_growth}")
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t")
        t1 = time.time()
        tb: Table = read_csv(ctx, rescaled_combined_single_drug_growth, csv_read_options)
        t2 = time.time()
        print(tb.shape, t2 - t1)
        print("Schema: ", tb.to_arrow().schema)
        tb['DOSE'] = -tb['LOG_CONCENTRATION']
        print("New Schema : ", tb.to_arrow().schema, tb.shape)
        columns = {'CELLNAME': 'CELL', 'DRUG_ID': 'DRUG', 'EXPID': 'STUDY'}
        tb.rename(columns)
        tb = tb[['SOURCE', 'CELL', 'DRUG', 'DOSE', 'GROWTH', 'STUDY']]
        print("Rename and Filter : ", tb.to_arrow().schema, tb.shape)
        print("----------------------------------------------")


load_single_dose_response_cylon()
