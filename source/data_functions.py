from typing import Optional

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler

def print_df_stats(df: pd.DataFrame, save_path=None) -> None:
    n_samples, n_features = df.shape
    task = 'Classification' if 'class' in df.columns else 'Regression'
    print(
        f'# Samples = {n_samples}\n'
        f'# Features = {n_features - 1}\n'
        f'Task: {task}\n'
    )
    print(df.head(3))
    if save_path is not None:
        pd.Series(
            {
                'n_samples': n_samples,
                'n_features': n_features,
                'task': task,
            }
        ).to_csv(save_path)

def load_prepare_data(
    data_path: str, 
    test_size: float, 
    split_seed: int, 
    verbose: bool = False,
    n_sample: Optional[int] = None,
) -> tuple[pd.DataFrame]:
    df = pd.read_csv(data_path, header=0)
    if n_sample:
        df = df.sample(n_sample, random_state=split_seed)
    if verbose: 
        print_df_stats(df)
    if 'class' in df.columns:
        target_col = 'class'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        raise ValueError("Bad task! Check columns for: 'class' or 'target'")
    return train_test_split(
        df[[v for v in df.columns if v != target_col]], 
        df[target_col], 
        test_size=test_size,
        random_state=split_seed,
    )

def scale_data(x, x_test, y, y_test) -> tuple[pd.DataFrame]:
    mms = MinMaxScaler()
    mms.fit(x)
    x, x_test = mms.transform(x), mms.transform(x_test)
    if y.dtype == float:
        y_mean, y_std = y.mean(), y.std()
        y, y_test = (y - y_mean) / y_std, (y_test - y_mean) / y_std
    return x, x_test, y, y_test
