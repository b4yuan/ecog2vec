from pynwb import NWBHDF5IO
import numpy as np
import soundfile as sf
import os
import scipy
from scipy.fft import fft, ifft, fftfreq, rfftfreq, rfft, irfft
from scipy.signal import butter, lfilter, filtfilt, hilbert
import re
from ripple2nwb.neural_processing import NeuralDataProcessor
from ripple2nwb.neural_processing import get_bipolar_referenced_electrodes
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
MCUs = MachineCompatibilityUtils()

class NeuralDataGenerator():
    def __init__(self, nwb_dir, patient):
        
        file_list = os.listdir(nwb_dir)
        self.nwb_dir = nwb_dir
        self.nwb_files = [file 
                          for file in file_list 
                          if file.startswith(f"{patient}")]
        self.sr = 0
        
        self.bad_electrodes = []
        self.good_electrodes = list(np.arange(256))
        
        self.high_gamma_min = 70
        self.high_gamma_max = 199

        self.electrode_name = '' # default for 401
        if patient != 'EFC402': 
            self.grid_size = np.array([16, 16])
        else:
            self.grid_size = np.array([8, 16])
        self.grid_step = 1

        self.elec_layout = np.arange(np.prod(
            self.grid_size)-1, -1, -1).reshape(self.grid_size).T[::self.grid_step, ::self.grid_step]

        self._bipolar_to_elec_map = None
        self._good_channels = None

    def bipolar_to_elec_map(self):
        # print('WARNING!!!!  MAKING UP bipolar_to_elec_map!!!')
        elec_map = []
        layout = self.elec_layout  # for short
        for i in range(layout.shape[0]):
            for j in range(layout.shape[1]):
                if j < layout.shape[1]-1:
                    elec_map.append((layout[i, j], layout[i, j+1]))
                if i < layout.shape[0]-1:
                    elec_map.append((layout[i, j], layout[i+1, j]))
        return np.array(elec_map)

    def good_channels(self):
        '''
        Pseudo-channels, constructed (on the fly) from the physical electrodes.
        For now at least, we won't USE_FIELD_POTENTIALS if we want to
        REFERENCE_BIPOLAR.

        NB!!: The *order* of these channels matters--it determines the order of
        the input data, and therefore is required by the functions that plot
        electrode_contributions in plotters.py! And the order of these channels
        will be determined by the *elec_layout*.
        '''

        # NB: this means that the electrodes are *not* in numerical order ('e1'
        #  does not correspond to the 0th entry in all_electrodes): as you can
        #  check, flattening the elec_layout does not yield an ordered list.
        all_electrodes = self.elec_layout.flatten().tolist()
        return [
            ch for ch, elec_pair in enumerate(self.bipolar_to_elec_map())
            if all([e in self.good_electrodes for e in elec_pair])
        ]
 
    def write_raw_data(self, 
                       chopped_sentence_dir=None,
                       sentence_dir=None,
                       chopped_recording_dir=None,
                       full_recording_dir=None,
                       chunk_length=100000,
                       block_list=None,
                       BPR=True):
        """
        Takes in an output directory and writes the speaking segments 
        of the ECoG data to that directory as multi-channel WAVE files.
        Can clip the data to a certain length (and throw out data less
        that that length) to handle batching.

        Args:
            output_dir (str): output directory to save WAVE files

        Returns:
            (None)
        """
        block_pattern = re.compile(r'B(\d+)')
        
        for file in self.nwb_files:
            
            match = block_pattern.search(file)
            block = int(match.group(1))
            
            if block_list and (block not in block_list):
                pass
            
            path = os.path.join(self.nwb_dir, file)
            
            io = NWBHDF5IO(path, load_namespaces=True, mode='r')
            nwbfile = io.read()

            try: 
                electrode_table = nwbfile.acquisition['ElectricalSeries'].\
                                        electrodes.table[:]
                self.nwb_sr = nwbfile.acquisition['ElectricalSeries'].\
                                rate
                if not BPR:        
                    # print('CAR...')       
                    indices = np.where(np.logical_or(
                        electrode_table['group_name'] == 
                        'L256GridElectrode electrodes', 
                        electrode_table['group_name'] == 
                        self.electrode_name) # Grid electrodes
                        )[0] # R256GridElectrode electrodes
                    
                    nwbfile_electrodes = nwbfile.acquisition['ElectricalSeries'].\
                                                data[:,indices]

                    nwbfile_electrodes = nwbfile_electrodes[:,self.good_electrodes] # only use good electrodes
                    print(f'Number of good electrodes in {file}: {nwbfile_electrodes.shape[1]}')
                    # assert nwbfile_electrodes.shape[1] == len(self.good_electrodes), \
                    #     f"Dimension issue..."

                    nwbfile_electrodes = notch_filter(nwbfile_electrodes, self.nwb_sr)
                    
                    w_l = self.high_gamma_min / (self.nwb_sr / 2) # Normalize the frequency
                    w_h = self.high_gamma_max / (self.nwb_sr / 2)
                    # b, a = butter(5, [w_l,w_h], 'band')
                    
                    # for ch in range(nwbfile_electrodes.shape[1]):
                    #     nwbfile_electrodes[:,ch] = filtfilt(b, 
                    #                                         a, 
                    #                                         nwbfile_electrodes[:,ch])
                    #     nwbfile_electrodes[:,ch] = np.abs(hilbert(nwbfile_electrodes[:,ch]))

                    # nwbfile_electrodes = common_average_reference(nwbfile_electrodes)
                    print(nwbfile_electrodes.shape)
                elif BPR:

                    print("Bipolar referencing...")
                    raw_data = nwbfile.acquisition['ElectricalSeries'].\
                                    data[()]
                    electrodes = nwbfile.acquisition['ElectricalSeries'].electrodes
                    nwbfile_electrodes, _, _ = get_bipolar_referenced_electrodes(
                        raw_data,
                        electrodes,
                        grid_size=self.grid_size
                    )
                    nwbfile_electrodes = nwbfile_electrodes[:,self._good_channels]
                    print(nwbfile_electrodes.shape)
                
                starts = list(nwbfile.trials[:]['start_time'] * self.nwb_sr)
                stops = list(nwbfile.trials[:]['stop_time'] * self.nwb_sr)
                
                ### Get speaking segments only ###
                starts = [int(start) for start in starts]
                stops = [int(stop) for stop in stops]
                
                i = 0
                all_speaking_segments = []
                for start, stop in zip(starts, stops):
                    speaking_segment = nwbfile_electrodes[start:stop,:]
                    all_speaking_segments.append(speaking_segment)
                            
                    if sentence_dir:
                        file_name = f'{sentence_dir}/{file}_{i}.wav'
                        sf.write(file_name, 
                                speaking_segment, 16000, subtype='FLOAT')
                    
                    i = i + 1
                    
                concatenated_speaking_segments = np.concatenate(all_speaking_segments, axis=0)
                
                if chopped_sentence_dir:
                    num_full_chunks = len(concatenated_speaking_segments) // chunk_length
                    # last_chunk_size = len(nwbfile_electrodes) % chunk_size

                    full_chunks = np.split(concatenated_speaking_segments[:num_full_chunks * chunk_length], num_full_chunks)
                    last_chunk = concatenated_speaking_segments[num_full_chunks * chunk_length:]

                    chunks = full_chunks # + [last_chunk] omit the last non-100000 chunk

                    # Loop through the chunks and save them as WAV files
                    for i, chunk in enumerate(chunks):
                        file_name = f'{chopped_sentence_dir}/{file}_{i}.wav' # CHANGE FOR EACH SUBJECT
                        sf.write(file_name, chunk, 16000, subtype='FLOAT')  # adjust as needed
                    
                if chopped_recording_dir:
                    
                    _nwbfile_electrodes = nwbfile_electrodes # [starts[0]:stops[-1],:] # remove starting/end silences
                    num_full_chunks = len(_nwbfile_electrodes) // chunk_length
                    # last_chunk_size = len(_nwbfile_electrodes) % chunk_size
                    
                    if num_full_chunks != 0:

                        full_chunks = np.split(_nwbfile_electrodes[:num_full_chunks * chunk_length], num_full_chunks)
                        last_chunk = _nwbfile_electrodes[num_full_chunks * chunk_length:]

                        chunks = full_chunks # + [last_chunk] omit the last non-100000 chunk
                        
                        # Checking lengths here
                        # for chunk in chunks:
                        #     print(chunk.shape)
                        # print(last_chunk.shape)

                        # Loop through the chunks and save them as WAV files
                        for i, chunk in enumerate(chunks):
                            file_name = f'{chopped_recording_dir}/{file}_{i}.wav' # CHANGE FOR EACH SUBJECT
                            sf.write(file_name, chunk, 16000, subtype='FLOAT')  # adjust as needed
                    
                if full_recording_dir:
                    file_name = f'{full_recording_dir}/{file}.wav'
                    sf.write(file_name, nwbfile_electrodes, 16000, subtype='FLOAT')

            except Exception as e: 
                print(f"An error occured and block {path} is not inluded in the wav2vec training data: {e}")
        
    def print(self):
        print('hi')

def common_average_reference(data):
    '''
    data:
        An ndarray (Nsamples x Nelectrodes) that has been processed
    '''

    return data - np.mean(data, axis=1, keepdims=True)

def notch_filter(
    X, f_sampling, f_notch=60, FILTER_HARMONICS=True, FFT=True
):
    # Author: Alex Bujan
    # Cribbed and lightly modified from ecogVIS by JGM

    f_nyquist = f_sampling/2
    if FILTER_HARMONICS:
        notches = np.arange(f_notch, f_nyquist, f_notch)
    else:
        notches = np.array([f_notch])

    if FFT:
        fs = rfftfreq(X.shape[0], 1/f_sampling)
        delta = 1.0
        fd = rfft(X, axis=0)
    else:
        n_taps = 1001
        gain = [1, 1, 0, 0, 1, 1]
    for notch in notches:
        if FFT:
            window_mask = np.logical_and(fs > notch-delta, fs < notch+delta)
            window_size = window_mask.sum()
            window = np.hamming(window_size)
            fd[window_mask, :] *= (1 - window)[:, None]
        else:
            f_critical = np.array(
                [0, notch-1, notch-.5, notch+.5, notch+1, f_nyquist]
            )/f_nyquist
            filt = scipy.signal.firwin2(n_taps, f_critical, gain)
            X = scipy.signal.filtfilt(filt, np.array([1]), X, axis=0)
    if FFT:
        X = irfft(fd, axis=0)
    return X