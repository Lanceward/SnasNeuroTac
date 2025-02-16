import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from spikingjelly.datasets import integrate_events_by_fixed_duration, play_frame, integrate_events_segment_to_frame



# threshold: minimum spikes in 1 timestep to activate pooled pixel
# max_time: maximum timesteps in the list allowed, exclusive
def threshold_array(arr, threshold, max_time):
    arr = arr[arr < max_time]
    uarr, counts = np.unique(arr, return_counts=True)
    counts[counts < threshold] = 0
    counts[counts >= threshold] = 1
    return np.repeat(uarr, counts)

def load_and_prepare_data(file_path, threshold = 1, max_timestamp=1000):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Pooling
    # crop to (480, 480)
    data = data[:, 80:560]
    
    h = data.shape[0]
    w = data.shape[1]
    # downsample to (40, 40)
    # reshape to (40, 40, 16)
    data = data.reshape(40, 12, 40, 12)
    data = data.swapaxes(1, 2)
    data = data.reshape(40, 40, 144)
    
    # Create an empty array with object dtype
    pooled_data = np.empty((40, 40), dtype=object)
            
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            concatarray = sum(data[i, j].tolist(), [])
            concatarray.sort()
            concatarray = np.array(concatarray)
            thresholded_concatarray = threshold_array(concatarray, threshold, max_timestamp)
            pooled_data[i, j] = thresholded_concatarray

    return pooled_data

def eventize_data(data):
    '''
    return dictionary with keys t, x, y, p
    '''
    total_datapoints_length = len(np.concatenate(data.flatten()))
    t = np.zeros(total_datapoints_length, dtype=int)
    x = np.zeros(total_datapoints_length, dtype=int)
    y = np.zeros(total_datapoints_length, dtype=int)
    p = np.ones(total_datapoints_length, dtype=int) # this will stay unchanged, as all pixels have positive polarity for our event camera

    idx = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pixel_event_len = len(data[i, j])
            t[idx : idx+pixel_event_len] = data[i, j]
            x[idx : idx+pixel_event_len] = i
            y[idx : idx+pixel_event_len] = j
            idx += pixel_event_len
    
    sorted_idx = np.argsort(t)
    t = t[sorted_idx]
    x = x[sorted_idx]
    y = y[sorted_idx]
    
    events = {}
    events['t'] = t
    events['x'] = x
    events['y'] = y
    events['p'] = p
    
    return events
    
def integrate_events_by_fixed_interval(events, interval, H, W):
    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']
    N = t.size

    frames = []
    current_interval = 0
    while True:
        left = np.where(t >= current_interval)[0][0]
        
        right_idxs = np.where(t > current_interval + interval)[0]
        if len(right_idxs) == 0:
            right = N-1
        else:
            right = right_idxs[0]
            
        frames.append(np.expand_dims(integrate_events_segment_to_frame(x, y, p, H, W, left, right), 0))

        current_interval += interval

        if current_interval > t[-1]:
            return np.concatenate(frames)

def play_single_texture_file(file_path):
    data = load_and_prepare_data(file_path, threshold = 2, max_timestamp=1000)
    events = eventize_data(data)
    frames = integrate_events_by_fixed_interval(events, 2, 40, 40)
    print(frames.shape)
    play_frame(frames, save_gif_to="frames.gif")

def ceiling_division(n, d):
    return -(n // -d)

class NeuroTacDataset(Dataset):
    def __init__(self, dataset_directory, unique_labels, interval = 1, max_timestamp = 1000):
        self.data_dir = dataset_directory
        self.unique_labels = unique_labels
        self.interval = interval
        self.max_timestamp = max_timestamp
        self.data = torch.tensor([])
        self.labels = torch.tensor([])
        self.Load_data()

    def Load_data(self):
        # pooling
        H = 40
        W = 40
        threshold = 2
        files = os.listdir(self.data_dir)
        data_length = len(files)
        data = np.zeros((ceiling_division(self.max_timestamp, self.interval), data_length, 2, H, W))
        labels = np.zeros(data_length)
        idx = 0
        for datapath in files:
            print("index " + str(idx) + ": " + datapath)
            #find label of datapath
            for i in range(len(self.unique_labels)):
                if self.unique_labels[i] in datapath:
                    labels[idx] = i
            
            #get data from datapath
            full_datapath = os.path.join(self.data_dir, datapath)
            d = load_and_prepare_data(full_datapath, threshold, max_timestamp=self.max_timestamp)
            e = eventize_data(d)
            f = integrate_events_by_fixed_interval(e, self.interval, H, W)
            #play_frame(f, save_gif_to="/Users/lanceshi/Desktop/DISS/code/SnasnetNeuroTac/DataSetGifs/" + datapath + ".gif")
            data[0:f.shape[0], idx, :, :, :] = f
            idx += 1
        
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
        print(self.data.shape)
        print(self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[:, idx, :, :, :], self.labels[idx]  # Return tuple (input, target)

if __name__ == "__main__":
    dir_path = "/Users/lanceshi/Desktop/DISS/code/Bristol/SNN/SlideS10"
    labels = ['acrylic', 'canvas', 'cotton', 'fashionfabric', 'felt', 'fur', 'mesh', 'nylon', 'wood', 'wool']
    nt_dataset = NeuroTacDataset(dir_path, labels)
    torch.save(nt_dataset, "NeuroTacDataset.pth")

