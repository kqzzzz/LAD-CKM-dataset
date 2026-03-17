from einops import rearrange
import numpy as np
import pandas as pd
from typing import List


class DynamicDatasetLoader:
    def __init__(self, dataset_file='datasets/Tx4x8_Rx2x2.npy',
                 norm=False, ratio=0.8, train=True) -> None:
        self.dataset = np.load(dataset_file, allow_pickle=True)
        self.dataset = [frame["csi"] for frame in self.dataset]                 # List of [n_rx, Nr, Nt, 2 * Nc]
        self.dataset = np.stack(self.dataset, axis=0)                           # [n_frame, n_rx, Nr, Nt, 2 * Nc]
        self.dataset = self.dataset.reshape(-1, *self.dataset.shape[2:])        # [n_frame * n_rx, Nr, Nt, 2 * Nc]
        self.dataset = rearrange(self.dataset, 'n r t c -> n c r t')            # [n_frame * n_rx, 2 * Nc, Nr, Nt]
        self.num = int(len(self.dataset) * ratio) if train else len(self.dataset) - int(len(self.dataset) * ratio)
        self.dataset = self.dataset[:self.num, :, :, :] if train else self.dataset[-self.num:, :, :, :]
        self.index = np.arange(0, self.num)
        (self.real_mean_ul,
         self.real_std_ul,
         self.imag_mean_ul,
         self.imag_std_ul,
         self.real_mean_dl,
         self.real_std_dl,
         self.imag_mean_dl,
         self.imag_std_dl) = self.get_stats()
        self.pl_values_ul = None  # Initialize pl_values for uplink
        self.pl_values_dl = None  # Initialize pl_values for downlink
        if norm:
            self.power_normalize_all()
            # self.cfr_normalize_all()
    
    def power_normalize_all(self):
        data_ul = self.dataset[:, :52, :, :]  # Uplink: first half
        data_dl = self.dataset[:, 52:, :, :]  # Downlink: second half
        
        self.pl_values_ul = np.zeros((data_ul.shape[0], 1, 1, 1), dtype=np.float32)
        self.pl_values_dl = np.zeros((data_dl.shape[0], 1, 1, 1), dtype=np.float32)
        
        for i in range(data_ul.shape[0]):
            pl_ul = np.sqrt(np.mean(np.abs(data_ul[i])**2, axis=(0, 1, 2), keepdims=True))
            data_ul[i] = data_ul[i] / pl_ul
            self.pl_values_ul[i] = pl_ul
            pl_dl = np.sqrt(np.mean(np.abs(data_dl[i])**2, axis=(0, 1, 2), keepdims=True))
            data_dl[i] = data_dl[i] / pl_dl
            self.pl_values_dl[i] = pl_dl
        
        self.dataset = np.concatenate((data_ul, data_dl), axis=1)
    
    def power_normalize(self, cfr, rx_ids, dl=True):
        if dl:
            pl = self.pl_values_dl[rx_ids]
        else:
            pl = self.pl_values_ul[rx_ids]
        norm_cfr = cfr / pl
        return norm_cfr
    
    def power_denormalize(self, norm_cfr, rx_ids, dl=True):
        if dl:
            pl = self.pl_values_dl[rx_ids]
        else:
            pl = self.pl_values_ul[rx_ids]
        return norm_cfr * pl
    
    def get_stats(self):
        data_ul = self.dataset[:, 0:52, :, :]
        data_dl = self.dataset[:, 52:104, :, :]
        real_ul, imag_ul = np.real(data_ul), np.imag(data_ul)
        real_dl, imag_dl = np.real(data_dl), np.imag(data_dl)
        real_mean_ul, real_std_ul, imag_mean_ul, imag_std_ul = real_ul.mean(), real_ul.std(), imag_ul.mean(), imag_ul.std()
        if np.isnan(real_dl).any() or np.isnan(imag_dl).any() or np.isinf(real_dl).any() or np.isinf(imag_dl).any():
            print("NaN or infinite values detected in the dataset.")
            nan_indices_real_dl = np.argwhere(np.isnan(real_dl) | np.isinf(real_dl))
        real_mean_dl, real_std_dl, imag_mean_dl, imag_std_dl = real_dl.mean(), real_dl.std(), imag_dl.mean(), imag_dl.std()
        return real_mean_ul, real_std_ul, imag_mean_ul, imag_std_ul, real_mean_dl, real_std_dl, imag_mean_dl, imag_std_dl

    def cfr_normalize_all(self):
        rx_ids = np.arange(self.num)
        data_ul = self.dataset[:, 0:52, :, :]
        data_dl = self.dataset[:, 52:104, :, :]
        norm_real_ul = (np.real(data_ul) - self.real_mean_ul) / self.real_std_ul
        norm_imag_ul = (np.imag(data_ul) - self.imag_mean_ul) / self.imag_std_ul
        norm_ul = norm_real_ul + 1j * norm_imag_ul
        norm_real_dl = (np.real(data_dl) - self.real_mean_dl) / self.real_std_dl
        norm_imag_dl = (np.imag(data_dl) - self.imag_mean_dl) / self.imag_std_dl
        norm_dl = norm_real_dl + 1j * norm_imag_dl
        for rx_id in rx_ids:
            self.dataset[rx_id, 0:52, :, :] = norm_ul[rx_id]
            self.dataset[rx_id, 52:104, :, :] = norm_dl[rx_id]
    
    def cfr_normalize(self, cfr, dl=True):  # cfr [batch_size, Nc, Nr, Nt]
        if dl:
            norm_real = (np.real(cfr) - self.real_mean_dl) / self.real_std_dl
            norm_imag = (np.imag(cfr) - self.imag_mean_dl) / self.imag_std_dl
        else:
            norm_real = (np.real(cfr) - self.real_mean_ul) / self.real_std_ul
            norm_imag = (np.imag(cfr) - self.imag_mean_ul) / self.imag_std_ul
        norm_cfr = norm_real + 1j * norm_imag
        return norm_cfr
    
    def cfr_restore(self, cfr, dl=True):  # cfr [batch_size, Nc, Nr, Nt]
        if dl:
            res_real = np.real(cfr) * self.real_std_dl + self.real_mean_dl
            res_imag = np.imag(cfr) * self.imag_std_dl + self.imag_mean_dl
        else:
            res_real = np.real(cfr) * self.real_std_ul + self.real_mean_ul
            res_imag = np.imag(cfr) * self.imag_std_ul + self.imag_mean_ul
        res_cfr = res_real + 1j * res_imag
        return res_cfr
    
    def get_downlink_cfr_batch(self, rx_ids):
        return self.dataset[rx_ids, 52:104, :, :]
    
    def get_uplink_cfr_batch(self, rx_ids):
        return self.dataset[rx_ids, 0:52, :, :]
    
    def get_freq(self, fc=6.7):
        up_c = fc + 0.015
        down_c = fc + 0.065
        num_elements = 52
        spacing = 3.125e-4
        up_fc = np.linspace(up_c - (num_elements // 2) * spacing, up_c + (num_elements // 2) * spacing, num_elements)
        down_fc = np.linspace(down_c - (num_elements // 2) * spacing, down_c + (num_elements // 2) * spacing, num_elements)
        fc = np.concatenate((up_fc, down_fc), axis=0).astype(np.float32)
        return fc
    
    def get_freq_center_down(self, fc=6.7):
        down_c = fc + 0.065
        return down_c
    
    def get_cfr_struct(self):
        shape = list(self.dataset.shape)
        shape[-3] = shape[-3] // 2
        return shape[-3:]


if __name__ == '__main__':
    
    dataset_file = f"datasets/Tx4x8_Rx2x2.npy"
    loader_train = DynamicDatasetLoader(dataset_file=dataset_file, train=True, ratio=0.8)
    loader_test = DynamicDatasetLoader(dataset_file=dataset_file, train=False, ratio=0.8)