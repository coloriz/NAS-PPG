from pathlib import Path
from textwrap import indent
from typing import Tuple, Optional

import numpy as np
from scipy.fft import rfft
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, resample_poly
from sklearn.preprocessing import scale


def gaussian(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.exp(-(((x - mean) / std) ** 2) / 2)


def envelope(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate upper and lower envelopes of the signal
    :param s: 1d signal
    :return: (upper_envelope, lower_envelope)
    """
    backward_diff = np.sign(s[1:-1] - s[:-2])
    forward_diff = np.sign(s[1:-1] - s[2:])

    u_x, = np.nonzero(np.logical_and(backward_diff > 0, forward_diff > 0))
    u_x = np.pad(u_x + 1, (1,), 'constant', constant_values=0)
    u_x[-1] = len(s) - 1
    upper = interp1d(u_x, s[u_x], 'linear')

    l_x, = np.nonzero(np.logical_and(backward_diff < 0, forward_diff < 0))
    l_x = np.pad(l_x + 1, (1,), 'constant', constant_values=0)
    l_x[-1] = len(s) - 1
    lower = interp1d(l_x, s[l_x], 'linear')

    return upper(np.arange(len(s))), lower(np.arange(len(s)))


def compute_envelope_amplitude(acc: np.ndarray) -> np.float:
    vals = []
    for Am_i in acc:
        upper, lower = envelope(Am_i)
        vals.append(np.mean(upper - lower))
    return np.mean(vals)


class Metadata:
    def __init__(self, sampling_rate: int, window_seconds: int, hopping_seconds: int):
        self.sampling_rate: int = sampling_rate
        self.window_seconds: int = window_seconds
        self.hopping_seconds: int = hopping_seconds
        self.window_size: int = window_seconds * sampling_rate
        self.hopping_size: int = hopping_seconds * sampling_rate


class Preprocessor:
    eps: np.float = np.finfo(np.float).eps

    def __init__(self,
                 metadata: Metadata,
                 bpf_cutoff: Tuple[float, float] = (0.4, 4),
                 upsample_factor: int = 1,
                 downsample_factor: int = 5,
                 fft_points: int = 2048,
                 hr_cutoff: Tuple[float, float] = (0.6, 3.3),
                 sigma: Optional[float] = None):
        self.metadata: Metadata = metadata
        self.bpf_cutoff: Tuple[float, float] = bpf_cutoff
        self.upsample_factor: int = upsample_factor
        self.downsample_factor: int = downsample_factor
        self.fft_points: int = fft_points
        self.hr_cutoff: Tuple[float, float] = hr_cutoff
        self.sigma: Optional[float] = sigma

        self.bpf_sos: np.ndarray = butter(4, bpf_cutoff, 'bandpass', output='sos', fs=metadata.sampling_rate)
        self.resampled_rate: float = metadata.sampling_rate * upsample_factor / downsample_factor
        self.frequency_resolution: float = self.resampled_rate / fft_points
        self.hr_cutoff_index = tuple(map(lambda x: round(x / self.frequency_resolution), hr_cutoff))
        self.output_size = self.hr_cutoff_index[1] - self.hr_cutoff_index[0] + 1

    def summary(self, prefix=''):
        def p(text):
            print(indent(text, prefix))
        p(f'sampling rate: {self.metadata.sampling_rate} Hz')
        p(f'window size: {self.metadata.window_seconds} s ({self.metadata.window_size})')
        p(f'hopping size: {self.metadata.hopping_seconds} s ({self.metadata.hopping_size})')
        p(f'BPF range: {self.bpf_cutoff} Hz')
        p(f'resampled rate: {self.resampled_rate} Hz')
        p(f'FFT points: {self.fft_points}')
        p(f'frequency resolution: {self.frequency_resolution:.4f} Hz ({self.frequency_resolution * 60:.4f} bpm)')
        p(f'HR cutoff range: {self.hr_cutoff} Hz {self.hr_cutoff_index}')
        p(f'sigma: {self.sigma}')
        p(f'output size: {self.output_size}')

    def frequency_bin_to_bpm(self, x: np.ndarray):
        """Convert frequency bin index back to bpm."""
        return np.float32((x * self.frequency_resolution + self.hr_cutoff[0]) * 60)

    def __call__(self,
                 ppg_signal: np.ndarray,
                 acc_signal: np.ndarray,
                 bpm: np.float) -> Tuple[np.ndarray, np.ndarray, np.float, np.ndarray]:
        """
        Apply preprocessing algorithm to given signals.
        :param ppg_signal: N x window_size PPG signals
        :param acc_signal: M x window_size Accelerometer signals
        :param bpm: True BPM
        :return: (PS_i, PA_i, Ia_i, y_true)
        """
        # Filter all signals using BPF
        Sn_i = sosfilt(self.bpf_sos, ppg_signal, axis=0)
        Am_i = sosfilt(self.bpf_sos, acc_signal, axis=0)

        # Standardize PPG signals and average them into a single signal
        Sn_i = scale(Sn_i, axis=1)
        S_i = np.mean(Sn_i, axis=0)

        # Resample each signal
        S_i = resample_poly(S_i, self.upsample_factor, self.downsample_factor)
        Am_i = resample_poly(Am_i, self.upsample_factor, self.downsample_factor, axis=1)

        # Compute the power spectra via a N-point FFT
        PS_i = np.abs(rfft(S_i, self.fft_points)) ** 2
        PAm_i = np.abs(rfft(Am_i, self.fft_points, axis=1)) ** 2

        # Normalize the power spectrum (0 ~ 1)
        PS_i = (PS_i - np.min(PS_i)) / np.ptp(PS_i)
        PAm_i = (PAm_i - np.min(PAm_i, axis=1)[..., np.newaxis]) / np.ptp(PAm_i, axis=1)[..., np.newaxis]

        # Average the power spectra from three-axis acceleration signals
        PA_i = np.mean(PAm_i, axis=0)

        # Extract only the possible HR range
        hr_PS_i = PS_i[self.hr_cutoff_index[0]:self.hr_cutoff_index[1] + 1]
        hr_PA_i = PA_i[self.hr_cutoff_index[0]:self.hr_cutoff_index[1] + 1]

        # Compute envelope amplitude of accelerometer signals
        acc_signal_resampled = resample_poly(acc_signal, self.upsample_factor, self.downsample_factor, axis=1)
        Ia_i = compute_envelope_amplitude(acc_signal_resampled)

        # Generate output vector
        bps = float(bpm) / 60
        frequency_bin_index = round(bps / self.frequency_resolution) - self.hr_cutoff_index[0]
        if self.sigma is None:
            y_true = np.zeros(self.output_size, np.float)
            y_true[frequency_bin_index] = 1
        else:
            x = np.linspace(self.hr_cutoff_index[0], self.hr_cutoff_index[1], self.output_size)
            y_true = gaussian(x, frequency_bin_index + self.hr_cutoff_index[0], self.sigma)
            y_true[y_true <= self.eps] = 0
            y_true /= np.sum(y_true)

        return hr_PS_i, hr_PA_i, Ia_i, y_true


processor_ISPC2015 = Preprocessor(
    metadata=Metadata(sampling_rate=125, window_seconds=8, hopping_seconds=2),
    bpf_cutoff=(0.4, 4),
    upsample_factor=1,
    downsample_factor=5,
    fft_points=2048,
    hr_cutoff=(0.6, 3.3),
    sigma=3)
processor_BAMI1 = processor_BAMI2 = Preprocessor(
    metadata=Metadata(sampling_rate=50, window_seconds=8, hopping_seconds=2),
    bpf_cutoff=(0.4, 4),
    upsample_factor=1,
    downsample_factor=2,
    fft_points=2048,
    hr_cutoff=(0.6, 3.3),
    sigma=3)


def preprocess_ISPC2015():
    dataset_path = Path(r'datasets/ISPC2015/')
    preprocessed_path = Path(r'preprocessed/ISPC2015/')
    preprocessed_path.mkdir(parents=True, exist_ok=True)
    # (signal, ground_truth, type, ppg_start, acc_start)
    ISPC2015 = [
        ('Training_data/DATA_01_TYPE01.mat', 'Training_data/DATA_01_TYPE01_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_02_TYPE02.mat', 'Training_data/DATA_02_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_03_TYPE02.mat', 'Training_data/DATA_03_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_04_TYPE02.mat', 'Training_data/DATA_04_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_05_TYPE02.mat', 'Training_data/DATA_05_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_06_TYPE02.mat', 'Training_data/DATA_06_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_07_TYPE02.mat', 'Training_data/DATA_07_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_08_TYPE02.mat', 'Training_data/DATA_08_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_09_TYPE02.mat', 'Training_data/DATA_09_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_10_TYPE02.mat', 'Training_data/DATA_10_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_11_TYPE02.mat', 'Training_data/DATA_11_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Training_data/DATA_12_TYPE02.mat', 'Training_data/DATA_12_TYPE02_BPMtrace.mat', 'T1', 1, 3),
        ('Extra_TrainingData/DATA_S04_T01.mat', 'Extra_TrainingData/BPM_S04_T01.mat', 'T1', 1, 3),
        ('TestData/TEST_S01_T01.mat', 'TrueBPM/True_S01_T01.mat', 'T2', 0, 2),
        ('TestData/TEST_S02_T01.mat', 'TrueBPM/True_S02_T01.mat', 'T2', 0, 2),
        ('TestData/TEST_S02_T02.mat', 'TrueBPM/True_S02_T02.mat', 'T3', 0, 2),
        ('TestData/TEST_S03_T02.mat', 'TrueBPM/True_S03_T02.mat', 'T3', 0, 2),
        ('TestData/TEST_S04_T02.mat', 'TrueBPM/True_S04_T02.mat', 'T3', 0, 2),
        ('TestData/TEST_S05_T02.mat', 'TrueBPM/True_S05_T02.mat', 'T3', 0, 2),
        ('TestData/TEST_S06_T01.mat', 'TrueBPM/True_S06_T01.mat', 'T2', 0, 2),
        ('TestData/TEST_S06_T02.mat', 'TrueBPM/True_S06_T02.mat', 'T3', 0, 2),
        ('TestData/TEST_S07_T02.mat', 'TrueBPM/True_S07_T02.mat', 'T3', 0, 2),
        ('TestData/TEST_S08_T01.mat', 'TrueBPM/True_S08_T01.mat', 'T2', 0, 2),
    ]

    print(' Preprocessing ISPC2015 '.center(30, '='))
    processor_ISPC2015.summary(' - ')
    metadata = processor_ISPC2015.metadata

    for subject_i, (data_path, bpm_path, group, ppg_index, acc_index) in enumerate(ISPC2015):
        data_mat = loadmat(str(dataset_path / data_path))['sig']
        bpm_mat = loadmat(str(dataset_path / bpm_path))['BPM0'].reshape((-1,))

        PS_list = []
        PA_list = []
        Ia_list = []
        y_true_list = []
        bpm_list = []

        for i, bpm in zip(range(0, data_mat.shape[1] - metadata.window_size + 1, metadata.hopping_size), bpm_mat):
            ppg_signal = data_mat[ppg_index:ppg_index + 2, i:i + metadata.window_size]
            acc_signal = data_mat[acc_index:acc_index + 3, i:i + metadata.window_size]

            PS_i, PA_i, Ia_i, y_true = processor_ISPC2015(ppg_signal, acc_signal, bpm)

            PS_list.append(PS_i)
            PA_list.append(PA_i)
            Ia_list.append(Ia_i)
            y_true_list.append(y_true)
            bpm_list.append(bpm)

        print(f'subject_{subject_i + 1:02d} - {len(PS_list)} samples')
        np.savez_compressed(
            preprocessed_path / f'subject_{subject_i + 1:02d}.npz',
            PS=np.array(PS_list, np.float32),
            PA=np.array(PA_list, np.float32),
            Ia=np.array(Ia_list, np.float32),
            y_true=np.array(y_true_list, np.float32),
            bpm=np.array(bpm_list, np.float32),
        )


def preprocess_BAMI_1():
    dataset_path = Path(r'datasets/BAMI-1/')
    preprocessed_path = Path(r'preprocessed/BAMI-1/')
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    print(' Preprocessing BAMI-I '.center(30, '='))
    processor_BAMI1.summary(' - ')
    metadata = processor_BAMI1.metadata

    for subject_i in range(1, 26):
        data_mat = loadmat(str(dataset_path / f'BAMI1_{subject_i}.mat'))
        raw_ppg = data_mat['rawPPG']
        raw_acc = data_mat['rawAcc'] / np.iinfo(np.uint16).max * 8 - 4
        bpm_mat = data_mat['bpm_ecg'].flatten()

        PS_list = []
        PA_list = []
        Ia_list = []
        y_true_list = []
        bpm_list = []

        for i, bpm in zip(range(0, raw_ppg.shape[1] - metadata.window_size + 1, metadata.hopping_size), bpm_mat):
            ppg_signal = raw_ppg[:, i:i + metadata.window_size]
            acc_signal = raw_acc[:, i:i + metadata.window_size]

            PS_i, PA_i, Ia_i, y_true = processor_BAMI1(ppg_signal, acc_signal, bpm)

            PS_list.append(PS_i)
            PA_list.append(PA_i)
            Ia_list.append(Ia_i)
            y_true_list.append(y_true)
            bpm_list.append(bpm)

        print(f'subject_{subject_i:02d} - {len(PS_list)} samples')
        np.savez_compressed(
            preprocessed_path / f'subject_{subject_i:02d}.npz',
            PS=np.array(PS_list, np.float32),
            PA=np.array(PA_list, np.float32),
            Ia=np.array(Ia_list, np.float32),
            y_true=np.array(y_true_list, np.float32),
            bpm=np.array(bpm_list, np.float32),
        )


def preprocess_BAMI_2():
    dataset_path = Path(r'datasets/BAMI-2/')
    preprocessed_path = Path(r'preprocessed/BAMI-2/')
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    print(' Preprocessing BAMI-II '.center(30, '='))
    processor_BAMI2.summary(' - ')
    metadata = processor_BAMI2.metadata

    for subject_i in range(1, 24):
        data_mat = loadmat(str(dataset_path / f'BAMI2_{subject_i}.mat'))
        raw_ppg = data_mat['rawPPG']
        raw_acc = data_mat['rawAcc'] / np.iinfo(np.uint16).max * 8 - 4
        bpm_mat = data_mat['bpm_ecg'].flatten()

        PS_list = []
        PA_list = []
        Ia_list = []
        y_true_list = []
        bpm_list = []

        for i, bpm in zip(range(0, raw_ppg.shape[1] - metadata.window_size + 1, metadata.hopping_size), bpm_mat):
            ppg_signal = raw_ppg[:, i:i + metadata.window_size]
            acc_signal = raw_acc[:, i:i + metadata.window_size]

            PS_i, PA_i, Ia_i, y_true = processor_BAMI2(ppg_signal, acc_signal, bpm)

            PS_list.append(PS_i)
            PA_list.append(PA_i)
            Ia_list.append(Ia_i)
            y_true_list.append(y_true)
            bpm_list.append(bpm)

        print(f'subject_{subject_i:02d} - {len(PS_list)} samples')
        np.savez_compressed(
            preprocessed_path / f'subject_{subject_i:02d}.npz',
            PS=np.array(PS_list, np.float32),
            PA=np.array(PA_list, np.float32),
            Ia=np.array(Ia_list, np.float32),
            y_true=np.array(y_true_list, np.float32),
            bpm=np.array(bpm_list, np.float32),
        )


if __name__ == '__main__':
    preprocess_ISPC2015()
    preprocess_BAMI_1()
    preprocess_BAMI_2()
