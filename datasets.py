import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from obspy import read
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_miniseed(file_path):
    stream = read(file_path)
    trace = stream[0]
    return trace

def compute_spectrogram(trace):
    trace_filtered = trace.copy()
    trace_filtered.filter('bandpass', freqmin=0.5, freqmax=1.0)
    freq, time, spectrogram = signal.spectrogram(trace_filtered.data, trace_filtered.stats.sampling_rate)
    return freq, time, spectrogram

def extract_features(trace, max_freq_bins, max_time_bins):
    _, time, spectrogram = compute_spectrogram(trace)
    normalized_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    padded_spectrogram = np.zeros((max_freq_bins, max_time_bins))
    padded_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = normalized_spectrogram
    features_tensor = torch.tensor(padded_spectrogram, dtype=torch.float32)
    return features_tensor, time

def plot_spectrogram(features, label, time_bins, title="Seismic Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(features, aspect='auto', origin='lower', cmap='jet')
    ax.set_title(title)
    ax.set_ylabel('Frequency Bin')
    ax.set_xlabel('Time Bin')

    arrival_index = np.argmax(label.numpy())
    arrival_time = time_bins[arrival_index] if arrival_index < len(time_bins) else None

    if arrival_time is not None:
        ax.axvline(x=arrival_index, color='red', linestyle='--', label=f'Arrival Time: {arrival_time:.2f}s')
        ax.legend()

    plt.show()

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


def train_test_split_dataset(dataset, test_size=0.2, random_state=42):
    # Get indices for the dataset
    indices = list(range(len(dataset)))
    # Split indices into train and test
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    # Create subsets for training and testing
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

class MarsSeismicDataset(Dataset):
    def __init__(self, data_dir, catalog_file, transform=None):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.mseed')]
        self.catalog = pd.read_csv(catalog_file)
        self.transform = transform
        self.max_freq_bins, self.max_time_bins, self.max_samples = self._get_max_shapes()

    def _get_max_shapes(self):
        max_freq_bins = 0
        max_time_bins = 0
        max_samples = 0
        for file_name in self.file_names:
            trace = read_miniseed(os.path.join(self.data_dir, file_name))
            _, time, spectrogram = compute_spectrogram(trace)
            max_freq_bins = max(max_freq_bins, spectrogram.shape[0])
            max_time_bins = max(max_time_bins, len(time))
            max_samples = max(max_samples, trace.stats.npts)
        return max_freq_bins, max_time_bins, max_samples

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        mseed_file = os.path.join(self.data_dir, self.file_names[idx])
        trace = read_miniseed(mseed_file)
        features, time_bins = extract_features(trace, self.max_freq_bins, self.max_time_bins)
        
        filename_base = os.path.splitext(os.path.basename(mseed_file))[0]
        catalog_filenames = self.catalog['filename'].apply(lambda x: os.path.splitext(x)[0])
        
        print(f"searching for filename in catalog: {filename_base}")

        if filename_base not in catalog_filenames.values:
            print("available filenames in catalog:", catalog_filenames.values)
        
        arrival_time_rel = self.catalog.loc[catalog_filenames == filename_base, 'time_rel(sec)'].values[0]
        arrival_index = np.searchsorted(time_bins, arrival_time_rel)
        
        label_array = np.zeros(self.max_time_bins)
        if arrival_index < self.max_time_bins:
            label_array[arrival_index] = 1
        
        label = torch.tensor(label_array, dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label, arrival_time_rel, time_bins

class LunarSeismicDataset(Dataset):
    def __init__(self, data_dir, catalog_file, transform=None, target_size=(129, 2555)):
        self.data_dir = data_dir
        self.file_names = []
        self.catalog = pd.read_csv(catalog_file)
        self.transform = transform
        self.mean, self.std = self._compute_dataset_statistics()
        self.target_size = target_size

    def _compute_dataset_statistics(self):
        total_sum = 0.0
        total_count = 0

        for file_name in os.listdir(self.data_dir):
            if not file_name.endswith('.mseed'):
                continue

            trace = read_miniseed(os.path.join(self.data_dir, file_name))
            _, time, spectrogram = compute_spectrogram(trace)

            # Filter spectrograms based on time bins
            time_bins = spectrogram.shape[1]
            if 1700 <= time_bins:
                self.file_names.append(file_name)
                total_sum += np.sum(spectrogram)
                total_count += spectrogram.size

        mean = total_sum / total_count

        total_squared_diff = 0.0
        for file_name in self.file_names:
            mseed_file = os.path.join(self.data_dir, file_name)
            trace = read_miniseed(mseed_file)
            _, time, spectrogram = compute_spectrogram(trace)

            total_squared_diff += np.sum((spectrogram - mean) ** 2)

        variance = total_squared_diff / total_count
        std = np.sqrt(variance)

        return mean, std

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        mseed_file = os.path.join(self.data_dir, file_name)
        trace = read_miniseed(mseed_file)
        _, time, spectrogram = compute_spectrogram(trace)

        # Standardize the spectrogram
        spectrogram = (spectrogram - self.mean) / self.std

        # Pad or truncate the spectrogram to the target size
        padded_spectrogram = np.zeros(self.target_size)
        padded_spectrogram[:, :spectrogram.shape[1]] = spectrogram[:, :self.target_size[1]]

        if self.transform:
            padded_spectrogram = self.transform(padded_spectrogram)

        # Convert to tensor
        padded_spectrogram = torch.tensor(padded_spectrogram, dtype=torch.float32)

        # Extract the label (arrival time as a float)
        filename_base = os.path.splitext(file_name)[0]
        arrival_time_rel = self.catalog.loc[self.catalog['filename'] == filename_base, 'time_rel(sec)']

        if arrival_time_rel.empty:
            # Skip this file if no catalog entry is found
            return None

        arrival_time_float = arrival_time_rel.values[0]

        return padded_spectrogram, torch.tensor(arrival_time_float, dtype=torch.float32)

class UnsupervisedLunarSeismicDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(129, 2555)):
        self.data_dir = data_dir
        self.file_names = []
        self.transform = transform
        self.mean, self.std = self._compute_dataset_statistics()
        self.target_size = target_size

    def _compute_dataset_statistics(self):
        total_sum = 0.0
        total_count = 0

        for file_name in os.listdir(self.data_dir):
            if not file_name.endswith('.mseed'):
                continue

            trace = read_miniseed(os.path.join(self.data_dir, file_name))
            _, time, spectrogram = compute_spectrogram(trace)

            # Filter spectrograms based on time bins
            time_bins = spectrogram.shape[1]
            if 1700 <= time_bins:
                self.file_names.append(file_name)
                total_sum += np.sum(spectrogram)
                total_count += spectrogram.size

        mean = total_sum / total_count

        total_squared_diff = 0.0
        for file_name in self.file_names:
            mseed_file = os.path.join(self.data_dir, file_name)
            trace = read_miniseed(mseed_file)
            _, time, spectrogram = compute_spectrogram(trace)

            total_squared_diff += np.sum((spectrogram - mean) ** 2)

        variance = total_squared_diff / total_count
        std = np.sqrt(variance)

        return mean, std

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        mseed_file = os.path.join(self.data_dir, file_name)
        trace = read_miniseed(mseed_file)
        _, time, spectrogram = compute_spectrogram(trace)

        # Standardize the spectrogram
        spectrogram = (spectrogram - self.mean) / self.std

        # Pad or truncate the spectrogram to the target size
        padded_spectrogram = np.zeros(self.target_size)
        padded_spectrogram[:, :spectrogram.shape[1]] = spectrogram[:, :self.target_size[1]]

        if self.transform:
            padded_spectrogram = self.transform(padded_spectrogram)

        # Convert to tensor
        padded_spectrogram = torch.tensor(padded_spectrogram, dtype=torch.float32)

        return padded_spectrogram



# class UnsupervisedLunarSeismicDataset(Dataset):
#     def __init__(self, data_dir, stats_file, transform=None, target_size=(129, 2555)):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.target_size = target_size

#         # Load precomputed statistics
#         self.mean, self.std = self.load_statistics(stats_file)

#         # Collect file names of precomputed spectrograms
#         self.file_names = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]

#     def load_statistics(self, stats_file):
#         # Load precomputed mean and std from a file
#         with open(stats_file, 'r') as f:
#             mean, std = map(float, f.readline().strip().split())
#         return mean, std

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         file_name = self.file_names[idx]
#         with h5py.File(os.path.join(self.data_dir, file_name), "r") as f:
#             spectrogram = f['spectrogram'][:]
        
#         # Standardize the spectrogram
#         spectrogram = (spectrogram - self.mean) / self.std

#         # Pad or truncate the spectrogram to the target size
#         padded_spectrogram = np.zeros(self.target_size)
#         padded_spectrogram[:, :spectrogram.shape[1]] = spectrogram[:, :self.target_size[1]]

#         if self.transform:
#             padded_spectrogram = self.transform(padded_spectrogram)

#         # Convert to tensor
#         padded_spectrogram = torch.tensor(padded_spectrogram, dtype=torch.float32)

#         return padded_spectrogram

# # To compute and save statistics and precompute spectrograms:
# def precompute_data(data_dir, output_dir, stats_file):
#     total_sum = 0.0
#     total_count = 0
#     file_names = [f for f in os.listdir(data_dir) if f.endswith('.mseed')]

#     for file_name in file_names:
#         trace = read_miniseed(os.path.join(data_dir, file_name))
#         _, time, spectrogram = compute_spectrogram(trace)

#         if spectrogram.shape[1] >= 1700:
#             with h5py.File(os.path.join(output_dir, file_name.replace('.mseed', '.h5')), "w") as f:
#                 f.create_dataset('spectrogram', data=spectrogram)

#             total_sum += np.sum(spectrogram)
#             total_count += spectrogram.size

#     mean = total_sum / total_count
#     total_squared_diff = 0.0

#     for file_name in file_names:
#         trace = read_miniseed(os.path.join(data_dir, file_name))
#         _, time, spectrogram = compute_spectrogram(trace)
#         total_squared_diff += np.sum((spectrogram - mean) ** 2)

#     variance = total_squared_diff / total_count
#     std = np.sqrt(variance)

#     # Save mean and std to a file
#     with open(stats_file, 'w') as f:
#         f.write(f"{mean} {std}")


