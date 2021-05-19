import os
import time
os.environ['MODIN_CPUS'] = "1"
import modin.pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_dataframe(dataframe: pd.DataFrame,
                    scaling_method: str, ):
    """new_df = dataframe_scaling(old_df, 'std')

    Scaling features in dataframe according to specific scaling strategy.
    TODO: More scaling options and selective feature(col) scaling for dataframe.

    Args:
        dataframe (pandas.Dataframe): dataframe to be scaled.
        scaling_method (str): 'std', 'minmax', etc.

    Returns:
        pandas.Dataframe: scaled dataframe.
    """
    print("=" * 80)
    print("scale_dataframe")
    scaling_method = scaling_method.lower()

    if scaling_method.lower() == 'none':
        return dataframe
    elif scaling_method.lower() == 'std':
        scaler = StandardScaler()
    elif scaling_method.lower() == 'minmax':
        scaler = MinMaxScaler()
    else:
        return dataframe

    if len(dataframe.shape) == 1:
        dataframe = scaler.fit_transform(dataframe.values.reshape(-1, 1))
    else:
        dataframe[:] = scaler.fit_transform(dataframe[:])
    print("=" * 80)
    return dataframe


t1 = time.time()
df = pd.read_csv(
    os.path.join(
        "/home/vibhatha/github/forks/Benchmarks/Data/Pilot1/raw/1/rescaled_combined_single_drug_growth_shuffle0"),
    sep=',',
    header=0,
    index_col=None, )
t2 = time.time()
df = df[['SOURCE', 'DRUG_ID', 'CELLNAME', 'LOG_CONCENTRATION', 'GROWTH']]
t3 = time.time()

# Delete '-', which could be inconsistent between seq and meta
df['CELLNAME'] = df['CELLNAME'].str.replace('-', '')
t4 = time.time()
# Encode data sources into numeric

# Scaling the growth with given scaling method
df['GROWTH'] = scale_dataframe(df['GROWTH'], "std")
t5 = time.time()
# Convert data type into generic python types
df[['LOG_CONCENTRATION', 'GROWTH']] = df[['LOG_CONCENTRATION', 'GROWTH']].astype(float)
t6 = time.time()

print("Time Taken For Data Loading : {} s".format(t2 - t1))
print("Time Taken For Column Select : {} s".format(t3 - t2))
print("Time Taken For Map Operation : {} s".format(t4 - t3))
print("Time Taken For Scale Operation : {} s".format(t5 - t4))
print("Time Taken For Casting : {} s".format(t6 - t5))

"""
Time Taken For Data Loading : 51.13017392158508 s
Time Taken For Column Select : 0.0009305477142333984 s
Time Taken For Map Operation : 0.013023853302001953 s
Time Taken For Scale Operation : 205.78279185295105 s
Time Taken For Casting : 39.518057346343994 s
"""