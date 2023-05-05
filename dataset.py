import numpy as np
import torch
from torch.utils.data import Dataset

class NonSeqDataSet(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data, self.label = self.load_all_npz_files(self.data_path)

    def load_one_npz_file(self, data_path):
        with np.load(data_path, allow_pickle=True) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels

    def load_all_npz_files(self, data_path_lists):
        all_data = []
        all_labels = []
        for tmp_path in data_path_lists:
            print('Loading {}...'.format(tmp_path))
            tmp_data, tmp_label = self.load_one_npz_file(tmp_path)
            tmp_data = tmp_data.astype(np.float32)
            all_data.append(tmp_data)
            tmp_label = tmp_label.astype(np.int64)
            all_labels.append(tmp_label)
        all_data = np.vstack(all_data).reshape(-1,1,3000)
        all_labels = np.hstack(all_labels)
        return all_data, all_labels


    def __getitem__(self, idx):
        sleep_data = self.data[idx]
        label = self.label[idx]
        return sleep_data, label

    def __len__(self):
        return len(self.label)



def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    data = np.squeeze(data)
    data = data[:, np.newaxis,:]
    return data, labels, sampling_rate

def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)
        data.append(tmp_data)
        labels.append(tmp_labels)
    return data, labels
class SeqDataset(Dataset):
    def __init__(self,data, labels, seq_len=20):
        self.x_list = []
        self.y_list = []
        for i in range(len(data)):

            data_len = data[i].shape[0]
            num_elems = (data_len//seq_len)*seq_len

            self.x_list.append(data[i][:num_elems])
            self.y_list.append(labels[i][:num_elems])

        self.x_list = [np.split(x, x.shape[0] // seq_len)
                       for x in self.x_list]
        self.y_list = [np.split(y, y.shape[0] // seq_len)
                       for y in self.y_list]

        self.x_list = [item for sublist in self.x_list for item in sublist]
        self.y_list = [item for sublist in self.y_list for item in sublist]

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.x_list[idx]), torch.LongTensor(self.y_list[idx]))

import glob
if __name__ == '__main__':
    all_data_path=glob.glob("./data/*.npz");
    dataset=NonSeqDataSet(all_data_path)
    for i,(data,label) in enumerate(dataset):
        print(data.shape)
        print(label)
