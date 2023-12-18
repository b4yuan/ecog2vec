from pynwb import NWBHDF5IO
import numpy as np
import soundfile as sf
import os
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter, filtfilt, hilbert
import re

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
        pass
    
    def write_hg_speaking_segments(self, output_dir, clipped=False, length=0):
        """
        Takes in an output directory and writes the speaking segments 
        of the ECoG data to that directory as multi-channel WAVE files.
        Can clip the data to a certain length (and throw out data less
        that that length) to handle batching.

        Args:
            output_dir (str): output directory to save WAVE files
            
        KW Args:
            clipped (bool): clip the WAVE files to a certain length
            length (int): length to clip WAVE files to

        Returns:
            (None)
        """
        
        for file in self.nwb_files:
            
            path = os.path.join(self.nwb_dir, file)
            
            io = NWBHDF5IO(path, load_namespaces=True, mode='r')
            nwbfile = io.read()

            nwbfile_electrodes = nwbfile.processing['ecephys'].\
                                         data_interfaces['LFP'].\
                                         electrical_series['preprocessed (bipolar)'].\
                                         data[:,self.good_electrodes]

            self.sr = nwbfile.processing['ecephys'].\
                              data_interfaces['LFP'].\
                              electrical_series['high gamma (bipolar)'].\
                              rate
                              
            w_l = self.high_gamma_min / (self.sr / 2) # Normalize the frequency
            w_h = self.high_gamma_max / (self.sr / 2)
            b, a = butter(5, [w_l,w_h], 'band')
            
            for ch in range(nwbfile_electrodes.shape[1]):
                nwbfile_electrodes[:,ch] = filtfilt(b, 
                                                    a, 
                                                    nwbfile_electrodes[:,ch])
            
            # calculate the analytic amplitude
            for ch in range(nwbfile_electrodes.shape[1]):
                analytic_signal = hilbert(nwbfile_electrodes[:,ch])
                nwbfile_electrodes[:,ch] = np.abs(analytic_signal)
            
            starts = list(nwbfile.trials[:]['start_time'] * self.sr)
            stops = list(nwbfile.trials[:]['stop_time'] * self.sr)
            
            ### Get speaking segments only ###
            starts = [int(start) for start in starts]
            stops = [int(stop) for stop in stops]
            
            i = 0
            for start, stop in zip(starts, stops):
                speaking_segment = nwbfile_electrodes[start:stop,:]
                if clipped == True: 
                    if speaking_segment.shape[0] > length:
                        file_name = f'{output_dir}/{file}_{i}.wav'
                        sf.write(file_name, 
                                 speaking_segment[0:length,:], 16000) 
                else:
                    file_name = f'{output_dir}/{file}_{i}.wav'
                    sf.write(file_name, 
                             speaking_segment, 16000)
                i = i + 1

            
    def write_raw_data(self, 
                       chopped_sentence_dir=None,
                       sentence_dir=None,
                       chopped_recording_dir=None,
                       full_recording_dir=None,
                       chunk_length=100000,
                       block_list=None):
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

            electrode_table = nwbfile.acquisition['ElectricalSeries'].\
                                      electrodes.table[:]
                                      
            indices = np.where(np.logical_or(electrode_table['group_name'] == 
                                             'L256GridElectrode electrodes', 
                                             electrode_table['group_name'] == 
                                             'R256GridElectrode electrodes'))[0]
            
            self.nwb_sr = nwbfile.acquisition['ElectricalSeries'].\
                              rate
            
            starts = list(nwbfile.trials[:]['start_time'] * self.nwb_sr)
            stops = list(nwbfile.trials[:]['stop_time'] * self.nwb_sr)

            nwbfile_electrodes = nwbfile.acquisition['ElectricalSeries'].\
                                         data[:,indices]
                                         
            nwbfile_electrodes = nwbfile_electrodes[:,self.good_electrodes] # only use good electrodes
                                         
            assert nwbfile_electrodes.shape[1] == 238, \
                f"Expected the second dimension to be 256, but got {nwbfile_electrodes.shape[1]}"
            
            w_l = self.high_gamma_min / (self.nwb_sr / 2) # Normalize the frequency
            w_h = self.high_gamma_max / (self.nwb_sr / 2)
            b, a = butter(5, [w_l,w_h], 'band')
            
            for ch in range(nwbfile_electrodes.shape[1]):
                nwbfile_electrodes[:,ch] = filtfilt(b, 
                                                    a, 
                                                    nwbfile_electrodes[:,ch])

                nwbfile_electrodes[:,ch] = np.abs(hilbert(nwbfile_electrodes[:,ch]))
              
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
        
    def print(self):
        print('hi')