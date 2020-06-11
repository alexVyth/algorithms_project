'''Main function to process dataset find commonn itemsets and
   save solution file for the defined problem.'''

import time

from mlxtend.frequent_patterns import fpmax
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


def read_csv(csv_dir, **kwargs):
    '''Read input dir as csv'''
    return pd.read_csv(csv_dir, delimiter=",", **kwargs)


def preprocess(dataframe):
    '''Preprocess data for less memory - cpu usage'''
    dataframe = dataframe.astype({"day": "int8", "hour": "int8",
                                  "ip": "int16", "mac": "int16"})
    dataframe = dataframe.sort_values(by=list(dataframe))
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def filter_data(dataframe):
    '''Filter data based based on time patterns'''
    return dataframe


def create_groups(dataframe):
    '''Create mac address groups from data,
    based on rows having same time, ip'''
    itemsets = list(dataframe.groupby(["day", "hour", "ip"])["mac"]
                    .apply(list).values)
    return [x for x in itemsets if len(x) > 1]


def encode_groups(data: list) -> pd.DataFrame:
    '''Take data list as input and return one-hot-encoded output'''
    encoder = TransactionEncoder()
    encoded_data_array = encoder.fit(data).transform(data)
    encoded_data_df = pd.DataFrame(encoded_data_array,
                                   columns=encoder.columns_)
    return encoded_data_df


def find_common(itemsets, occurrences, **kwargs):
    '''Find common itemsets with FPGroth algorithm'''
    min_support = occurrences / len(itemsets)
    return fpmax(itemsets, min_support=min_support, **kwargs)


def postprocess(dataframe):
    '''Filter results based on defined options'''
    # dataframe['length'] = dataframe['itemsets'].apply(len)
    # dataframe = dataframe[dataframe['length'] >= 3]
    return dataframe


def output(data):
    '''Create solution output file'''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    csv_out = f"./runs/{timestr}.csv"
    return csv_out, data


if __name__ == "__main__":
    # Input file
    CSV = "./examples/recordings_example.csv"
    # CSV = "./recordings/recordings.csv"
    df = read_csv(CSV, names=["day", "hour", "ip", "mac"])

    # Main Process
    df = preprocess(df)
    df = filter_data(df)
    groups = create_groups(df)
    groups = encode_groups(groups)
    common = find_common(groups, occurrences=5, use_colnames=True)
    common = postprocess(common)
    with pd.option_context('display.max_rows', None):
        print(common)
    output(common)
