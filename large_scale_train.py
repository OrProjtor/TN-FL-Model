import os
import json
import argparse
from time import time
from pathlib import Path
from collections import defaultdict
from typing import Optional, Callable

import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RepeatedKFold


from source.models.QCPR import QCPR
from source.models.QCPRf import QCPRf
from source.features import FFeature
from source.data_functions import load_prepare_data

REG_TASK = 'regression'
CLS_TASK = 'classification'
DATA_DIR = Path('./data')

def load_transform_data(
    data_path, 
    test_size, 
    split_seed, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    n_sample: Optional[int] = None
):
    x, x_test, y, y_test = load_prepare_data(data_path, test_size, split_seed, n_sample=n_sample)
    if transform_x: 
        mms = MinMaxScaler()
        mms.fit(x)
        x, x_test = mms.transform(x), mms.transform(x_test)  
    if transform_y:
        y_mean, y_std = y.mean(), y.std()
        y, y_test = (y - y_mean) / y_std, (y_test - y_mean) / y_std
    return x, x_test, y, y_test

def create_dir_if_not_exists(directory: str) -> os.PathLike:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def model_factory(model_cls, params, task: str = REG_TASK):
    tasks = [REG_TASK, CLS_TASK]
    if task not in tasks:
        raise ValueError(f'Bad task: {task}. Need one of these: {tasks}')
    return Pipeline([('scaler', MinMaxScaler()), (task, model_cls(**params))])

def prepare_train_funct_fl(model_params):
    def train_model(x, y):
        model = model_factory(QCPRf, model_params, REG_TASK) 
        model.fit(x, y)  
        return model, 'FL_Model'
    return train_model

def prepare_train_funct_cv(model_params):
    def train_model(x, y):
        all_scores = []
        for value in model_params['features_list']:
            params = dict(feature_map=value) | model_params['fixed_params']
            model = model_factory(QCPR, params, REG_TASK)
            cv = RepeatedKFold(
                n_splits=model_params['n_splits'],
                n_repeats=model_params['n_repeats'], 
                random_state=model_params['kfold_rs']
            )
            scores = cross_val_score(
                model, x, y, scoring='neg_mean_squared_error', cv=cv, 
                n_jobs=model_params['n_jobs'])
            all_scores.append(np.mean(scores))

        best_feature = model_params['features_list'][np.argmax(all_scores)]
        params = dict(feature_map=best_feature) | model_params['fixed_params']
        model = model_factory(QCPR, params, REG_TASK)
        model.fit(x, y)  
        return model, 'CV_Model'
    return train_model

def update_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].append(value)

def extend_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].extend(value.copy())

def train_model_time(train_model: Callable, x: np.ndarray, y: np.ndarray) -> dict:
    start_t = time()
    model, model_name = train_model(x, y)
    end_time = time() - start_t
    return dict(model=model, model_name=model_name, train_time=end_time)

def get_stats_several_trials(
    data_path: str,
    train_model: Callable,
    metric_f: Callable,
    n_trials: int = 10,
    test_size: float = 0.2,
    tqdm_disable: bool = False,
    transform_x: bool = False,
    transform_y: bool = True,
    n_sample: Optional[int] = None
):
    res = defaultdict(list)
    for trial in tqdm(range(1, n_trials + 1), disable=tqdm_disable): # Do several trials to get mean and std of metrics
        x, x_test, y, y_test = load_transform_data(
            data_path, test_size, trial, transform_x, transform_y, n_sample)
        model_dt = train_model_time(train_model, x, y)
        # Compute stats:
        final_score = metric_f(y_test, model_dt['model'].predict(x_test))
        update_results_dict(
            res, trial=trial, model=model_dt['model_name'], 
            metric=final_score, train_time=model_dt['train_time'],
        )
    return res

def data2model_params(data_name: str, model_name: str, p_scale_list: list) -> dict:
    fixed_p = dict(n_epoch=10, alpha=1e-2, random_state=13)
    if data_name == 'airline':
        data_p = dict(rank=20, m_order=64) 
    else:
        raise ValueError(f'Bad data name: {data_name}')

    if model_name == 'FL_Model':
        model_p = dict(fmaps_list=list(map(FFeature, p_scale_list)),
            beta=1e-1, lambda_reg_type='l1', n_steps_l1=500, update_order_t='lw')
        return fixed_p | data_p | model_p
    elif model_name == 'CV_Model':
        model_params = dict(features_list=list(map(FFeature, p_scale_list)),
            n_jobs=-1, n_splits=3, n_repeats=2, kfold_rs=1,
            fixed_params=fixed_p | data_p,
        )
        return model_params

def data2train_f(data_name: str, model_name: str, p_scale_list: list) -> Callable:
    if model_name == 'FL_Model':
        return prepare_train_funct_fl(data2model_params(data_name, model_name, p_scale_list))
    elif model_name == 'CV_Model':
        return prepare_train_funct_cv(data2model_params(data_name, model_name, p_scale_list))
    
def prepare_for_dump(dt):
    for k in dt.keys():
        if isinstance(dt[k][0], np.ndarray):
            dt[k] = [list(v) for v in dt[k]]
    return dt

N_TRIALS = 5
P_SCALE_LIST = [10, 2, 128, 25, 64, 1024]
METRIC_F = mean_squared_error 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment settings")
    # Add arguments
    parser.add_argument('data', type=str, help='dataset name')
    parser.add_argument('--n_sample', type=int, default=None, help='the number of samples to take from the data')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--save_name', type=str, default='results')
    # Parse the arguments
    args = parser.parse_args()
    DATA_NAME, N_SAMPLE, TEST_SIZE, SAVE_NAME = args.data, args.n_sample, args.test_size, args.save_name

    DATA_PATH = DATA_DIR / f'{DATA_NAME}.csv'
    RES_PATH = create_dir_if_not_exists(f'./artifacts/FLvsCV_temp/{DATA_NAME}') / f'{SAVE_NAME}.json'

    print(DATA_NAME, N_SAMPLE, TEST_SIZE)
    
    res = defaultdict(list)
    for model_name in ['FL_Model', 'CV_Model']:
        for nps in range(1, len(P_SCALE_LIST) + 1):
            p_scale_list = P_SCALE_LIST[:nps]
            train_model_f = data2train_f(DATA_NAME, model_name, p_scale_list)
            _res = get_stats_several_trials(DATA_PATH, train_model_f, METRIC_F, N_TRIALS, TEST_SIZE, n_sample=N_SAMPLE)
            extend_results_dict(res, n_features=[nps,]*N_TRIALS, **_res)
    res = prepare_for_dump(res)
    with open(RES_PATH, 'w') as outfile:
        json.dump(res, outfile)
