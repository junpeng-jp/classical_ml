import pandas as pd
import numpy as np
import pandas.api.types as pd_types

def pd_summary(df):
    # sample a maximum of 100k rows randomly
    if len(df) > 100000:
        df = df.sample(100000)

    # dtypes
    stats = {}
    for col in df.columns:
        if pd_types.is_object_dtype(df[col]) or pd_types.is_categorical_dtype(df[col]):
            stats[col] = df[col].value_counts()
            stats[col][np.nan] = df[col].isna().sum()            

        elif pd_types.is_numeric_dtype(df[col]):
            stats[col] = df[col].quantile([0, 0.25, 0.5, 0.75, 1])
            stats[col].index = ['0%','25%', '50%', '75%', '100%']
            stats[col]['mean'] = df[col].mean()
            stats[col][np.nan] = df[col].isna().sum()
            stats[col] = stats[col][['0%', '25%', '50%', 'mean', '75%', '100%', np.nan]]
        else:
            stats[col] = None

        print("----- {} -----".format(col))
        print(stats[col])
        print("\n")

    

