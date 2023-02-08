import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle


def shift_s(s, shift_arr):
    df = np.empty(shape=(len(s), len(shift_arr)), dtype=np.float32)
    col_names = []

    for i, shift in enumerate(shift_arr):
        df[:, i] = s.shift(shift).values
        col_names.append(str(s.name) + "_" + str(shift))   

    df = pd.DataFrame(df, index=s.index, columns=col_names)
    return df


def get_pickle_from_s3(file_path, bucket_name="eles-ocenjevalnik"):
    s3client = boto3.client('s3')
    response = s3client.get_object(Bucket=bucket_name, Key=file_path)
    body = response['Body']

    data = pickle.loads(body.read())
    return data

def upload_pickle_to_s3(df, file_path, bucket_name="eles-ocenjevalnik"):
    s3_resource = boto3.resource('s3')
    buffer = pickle.dumps(df)
    s3_resource.Object(bucket_name, file_path).put(Body=buffer);