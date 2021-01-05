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


def load_aggregated_single_response_pandas(target='AUC', min_r2_fit=0.3, max_ec50_se=3.0,
                                           combo_format=True,
                                           rename=False):
    url = "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_single_response_agg"
    output_combined_single_response = \
        "/home/vibhatha/data/uno/Pilot1/workload_1/combined_single_response_agg"

    if not os.path.exists(output_combined_single_response):
        download_util.download(url=url, output_file=output_combined_single_response)

    if os.path.exists(output_combined_single_response):
        print(f"Pandas Data file : {output_combined_single_response}")
        t1 = time.time()
        df = pd.read_csv(output_combined_single_response, engine='c', sep='\t',
                         dtype={'SOURCE': str, 'CELL': str, 'DRUG': str, 'STUDY': str,
                                'AUC': np.float32, 'IC50': np.float32,
                                'EC50': np.float32, 'EC50se': np.float32,
                                'R2fit': np.float32, 'Einf': np.float32,
                                'HS': np.float32, 'AAC1': np.float32,
                                'AUC1': np.float32, 'DSS1': np.float32})
        t2 = time.time()
        df = df[(df['R2fit'] >= min_r2_fit) & (df['EC50se'] <= max_ec50_se)]
        filter_time = time.time() - t2
        print("Pandas Data Loading Time ", df.shape, t2 - t1)
        print("Pandas Filter Time 1", df.shape, filter_time)
        df = df[['SOURCE', 'CELL', 'DRUG', target, 'STUDY']]
        df = df[~df[target].isnull()]
        print("After not and null check ", df.shape)

        if combo_format:
            df = df.rename(columns={'DRUG': 'DRUG1'})
            df['DRUG2'] = np.nan
            df['DRUG2'] = df['DRUG2'].astype(object)
            df = df[['SOURCE', 'CELL', 'DRUG1', 'DRUG2', target, 'STUDY']]
            if rename:
                df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                        'DRUG1': 'Drug1', 'DRUG2': 'Drug2', 'STUDY': 'Study'})
        else:
            if rename:
                df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                        'DRUG': 'Drug', 'STUDY': 'Study'})

        print("DF New", df.shape, df.columns)


def load_single_dose_response_pandas(combo_format=False, fraction=True):
    url = "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/rescaled_combined_single_drug_growth"
    rescaled_combined_single_drug_growth = \
        "/home/vibhatha/data/uno/Pilot1/workload_1/rescaled_combined_single_drug_growth"

    if not os.path.exists(rescaled_combined_single_drug_growth):
        download_util.download(url=url, output_file=rescaled_combined_single_drug_growth)
    if os.path.exists(rescaled_combined_single_drug_growth):
        print(f"Data file : {rescaled_combined_single_drug_growth}")
        print("------------------Pandas--------------------")
        t1 = time.time()
        df = pd.read_csv(rescaled_combined_single_drug_growth, sep='\t', engine='c',
                         na_values=['na', '-', ''],
                         # nrows=10,
                         dtype={'SOURCE': str, 'DRUG_ID': str,
                                'CELLNAME': str, 'CONCUNIT': str,
                                'LOG_CONCENTRATION': np.float32,
                                'EXPID': str, 'GROWTH': np.float32})
        t2 = time.time()
        print(df.shape, t2 - t1)
        print("Schema : ", df.dtypes, df.shape)
        df['DOSE'] = -df['LOG_CONCENTRATION']
        print("New Schema : ", df.dtypes, df.shape)
        df = df.rename(columns={'CELLNAME': 'CELL', 'DRUG_ID': 'DRUG', 'EXPID': 'STUDY'})
        df = df[['SOURCE', 'CELL', 'DRUG', 'DOSE', 'GROWTH', 'STUDY']]
        print("Rename and Update : ", df.dtypes, df.shape)
        print("----------------------------------------------")


load_single_dose_response_pandas()
