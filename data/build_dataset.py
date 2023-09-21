import numpy as np
import pandas as pd
import h5py
import os

from tqdm.notebook import tqdm 
from glob import glob
import requests
from multiprocessing import Pool
import time


data_dir = "/mnt/home/wwong/ceph/Dataset/GW/DiffusionLikelihood/O1/H1"

def read_data(path, **kws):
    with h5py.File(path, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
        dt = T/len(h)
        time = t0 + dt*np.arange(len(h))
        return pd.Series(h, index=time, **kws)

def split_segment(data, T = 4):
    data_segments = []
    dt = data.index[1] - data.index[0]
    # segment length
    N = int(round(T / dt))
    # number of segments
    N_segments = int(len(data) / N)
    data_segments += [data.iloc[k*N:k*N+N] for k in range(N_segments)]
    return data_segments

def filter_segment(data_segments):
    good_segments = []
    for s in data_segments:
        no_events = all([(t0 < s.index[0] or t0 > s.index[-1]) for t0 in true_event_times])
        no_nans = not s.isnull().values.any()
        if no_events and no_nans:
            good_segments.append(s)
    return good_segments

url = 'https://www.gw-openscience.org/eventapi/jsonfull/GWTC/'
with requests.get(url) as r:
    rjson = r.json()

true_event_times = sorted([v['GPS'] for v in rjson['events'].values()])

def process_data(path):
    # print(path)
    data = read_data(path)
    data_segments = split_segment(data)
    good_segments = filter_segment(data_segments)
    array = np.array([s.values for s in good_segments])
    return array

def multiprocess_data(path_list):
    with Pool(24) as p:
        tensor = p.map(process_data, path_list)
        p.close()
    return np.concatenate([i if i.any() else np.array([]).reshape(0,16384) for i in tensor])

if __name__=='__main__':

    path_list = sorted(glob(os.path.join(data_dir, '*.hdf5')))
    print(len(path_list))
    current_time = time.time()
    result = []
    chunk_size = 24
    for i in range(0, len(path_list), chunk_size):
        print(i, flush=True)
        result.append(multiprocess_data(path_list[i:i+chunk_size]))
        
    result = np.concatenate(result)
    print("Time taken: ", time.time() - current_time)
    scale = np.mean(np.abs(result[::1000]))
    std = np.std(result[::1000])
    with h5py.File('/mnt/home/wwong/ceph/Dataset/GW/DiffusionLikelihood/O1/H1_processed.hdf5', 'w') as f:
        f.create_dataset('data', data=result)
        f.attrs['scale'] = scale
        f.attrs['std'] = std
