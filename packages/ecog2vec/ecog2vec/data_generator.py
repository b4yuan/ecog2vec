from pynwb import NWBHDF5IO
import numpy as np
import soundfile as sf
import os
import scipy
from scipy.fft import fft, ifft, fftfreq, rfftfreq, rfft, irfft
from scipy.signal import butter, lfilter, filtfilt, hilbert
import re
# from ripple2nwb.neural_processing import NeuralDataProcessor
# from prepype import NeuralDataProcessor
from prepype.neural_processing import NeuralDataProcessor, downsample, downsample_NWB
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
from utils_jgm.toolbox import resample
import tensorflow as tf
from machine_learning.neural_networks import tf_helpers as tfh
MCUs = MachineCompatibilityUtils()

class NeuralDataGenerator():


    '''
    Usage:

    
        from ecog2vec.data_generator import NeuralDataGenerator

        nwb_dir = 'path_to_nwb_files/EFC401'

        patient_b = NeuralDataGenerator(nwb_dir, 'EFC401')

        patient_b.make_data(
            chopped_sentence_dir='path_to_data_dir/chopped_sentence',
            sentence_dir='path_to_data_dir/sentence',
            chopped_recording_dir='path_to_data_dir/chopped_recording',
            full_recording_dir='path_to_data_dir/full_recording',
            ecog_tfrecords_dir='path_to_data_dir/ecog_tfrecords',
            chunk_length=100000,
            BPR=False)

    '''


    def __init__(self, nwb_dir, patient):

        self.patient = patient
        
        file_list = os.listdir(nwb_dir)
        self.nwb_dir = nwb_dir
        self.nwb_files = [file 
                          for file in file_list 
                          if file.startswith(f"{patient}")]
        self.target_sr = 100
        
        self.bad_electrodes = []
        self.good_electrodes = list(np.arange(256))
        
        self.high_gamma_min = 70
        self.high_gamma_max = 199

        # Bad electrodes are 1-indexed!
        
        if patient == 'EFC400':
            self.electrode_name = 'R256GridElectrode electrodes'
            self.grid_size = np.array([16, 16])
            self.bad_electrodes = [x - 1 for x in [1, 2, 33, 50, 54, 64, 
                                                   128, 129, 193, 194, 256]]
            self.blocks_ID_mocha = [3, 23, 72]
            
        elif patient == 'EFC401':
            self.electrode_name = 'L256GridElectrode electrodes'
            self.grid_size = np.array([16, 16])
            self.bad_electrodes = [x - 1 for x in [1, 2, 63, 64, 65, 127, 
                                                   143, 193, 194, 195, 196, 
                                                   235, 239, 243, 252, 254, 
                                                   255, 256]]
            self.blocks_ID_mocha = [4, 41, 57, 61, 66, 69, 73, 77, 83, 87]

        elif patient == "EFC402":
            self.electrode_name = 'InferiorGrid electrodes'
            self.grid_size = np.array([8, 16])
            self.bad_electrodes = [x - 1 for x in list(range(129, 257))]
            self.blocks_ID_demo2 = [4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 
                                    18, 19, 25, 26, 27, 33, 34, 35, 44, 45, 
                                    46, 47, 48, 49, 58, 59, 60]
            
        elif patient == 'EFC403':
            self.electrode_name = 'Grid electrodes'
            self.grid_size = np.array([16, 16])
            self.bad_electrodes = [x - 1 for x in [129, 130, 131, 132, 133, 
                                                   134, 135, 136, 137, 138,
                                                   139, 140, 141, 142, 143, 
                                                   144, 145, 146, 147, 148,
                                                   149, 161, 162, 163, 164, 
                                                   165, 166, 167, 168, 169,
                                                   170, 171, 172, 173, 174, 
                                                   175, 176, 177, 178, 179,
                                                   180, 181]]
            self.blocks_ID_demo2 = [4, 7, 10, 13, 19, 20, 21, 28, 35, 39, 
                                    52, 53, 54, 55, 56, 59, 60, 61, 62, 63, 
                                    64, 70, 73, 74, 75, 76, 77, 83, 92, 93, 
                                    94, 95, 97, 98, 99, 100, 101, 108, 109, 
                                    110, 111, 112, 113, 114, 115]
            
        else:
            self.electrode_name = None
            self.grid_size = None
            self.bad_electrodes = None

        self.good_electrodes = [x for x in self.good_electrodes if x not in self.bad_electrodes]

        self.config = None

 
    def make_data(self, 
                    chopped_sentence_dir=None,
                    sentence_dir=None,
                    chopped_recording_dir=None,
                    full_recording_dir=None,
                    ecog_tfrecords_dir=None,
                    chunk_length=100000,
                    BPR=None):
        """
        Takes in an output directory and writes the speaking segments 
        of the ECoG data to that directory as multi-channel WAVE files.
        Can clip the data to a certain length (and throw out data less
        that that length) to handle batching.

        Args:
            *_dir (str): output directory to save WAVE files for training
                         and extracting features.
            chunk_length (int): length of each chunk of data to save.
            BPR (bool): whether to use bipolar referencing or CAR

        Returns:
            (None)
        """
        all_example_dict = [] # not maintained at the moment; stores ALL example dicts
        
        block_pattern = re.compile(r'B(\d+)')

        if BPR is None:
            raise ValueError("Please specify whether to use common average reference or bipolar referencing")

        if self.config is None:
            self.config = {
                'referencing': {'CAR'} if not BPR else {'bipolar'},
                'notch filter': 60.0,
                'target sampling rate': None,
                'grid size': self.grid_size
            }
        
        for file in self.nwb_files:

            create_training_data = True
            
            match = block_pattern.search(file)
            block = int(match.group(1))
            
            if self.patient == 'EFC400' or self.patient == 'EFC401':
                if block in self.blocks_ID_mocha:
                    create_training_data = False
            elif self.patient == 'EFC402' or self.patient == 'EFC403':
                if block in self.blocks_ID_demo2:
                    create_training_data = False
            
            path = os.path.join(self.nwb_dir, file)
            
            io = NWBHDF5IO(path, load_namespaces=True, mode='r')
            nwbfile = io.read()

            try:

                with NeuralDataProcessor(
                    nwb_path=path, config=self.config, WRITE=False
                ) as processor:
                        
                    # Grab the electrode table and sampling rate,
                    # and then process the raw ECoG data. 

                    electrode_table = nwbfile.acquisition["ElectricalSeries"].\
                                            electrodes.table[:]
                    
                    self.nwb_sr = nwbfile.acquisition["ElectricalSeries"].\
                                    rate

                    # indices = np.where(electrode_table["group_name"] == 
                    #                    self.electrode_name
                    #                    )[0]

                    print(f'Referencing with {self.config["referencing"]}...')

                    processor.preprocess()
                    print('Preprocessing done.')

                    processor.edwards_high_gamma()
                    print('High gamma extraction done.')

                    nwbfile_electrodes = processor.nwb_file.processing['ecephys'].\
                                                    data_interfaces['LFP'].\
                                                    electrical_series[f'high gamma ({list(self.config["referencing"])[0]})'].\
                                                    data[()][:, self.good_electrodes]

                    print(f"Number of good electrodes in {file}: {nwbfile_electrodes.shape[1]}")                       
                    
                    # Begin building the WAVE files for wav2vec training
                    # and evaluation.

                    # Starts/stops for each intrablock trial.
                    starts = [int(start) 
                              for start 
                              in list(nwbfile.trials[:]["start_time"] * self.nwb_sr)]
                    stops = [int(start)
                             for start
                             in list(nwbfile.trials[:]["stop_time"] * self.nwb_sr)]
                    
                    # Manage the speaking segments only... as an option .
                    # Training data for wav2vec as speaking segments only
                    # will be saved in the `chopped_sentence_dir` directory. 
                    # This block also saves the individual sentences.                 
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
                    
                    # Training data: speaking segments only
                    if create_training_data and chopped_sentence_dir:
                        num_full_chunks = len(concatenated_speaking_segments) // chunk_length
                        # last_chunk_size = len(nwbfile_electrodes) % chunk_size

                        full_chunks = np.split(concatenated_speaking_segments[:num_full_chunks * chunk_length], num_full_chunks)
                        last_chunk = concatenated_speaking_segments[num_full_chunks * chunk_length:]

                        chunks = full_chunks # + [last_chunk] omit the last non-100000 chunk

                        # Loop through the chunks and save them as WAV files
                        for i, chunk in enumerate(chunks):
                            file_name = f'{chopped_sentence_dir}/{file}_{i}.wav'
                            sf.write(file_name, chunk, 16000, subtype='FLOAT')

                        print(f'Out of distribution block. Number of chopped chunks w/o intertrial silences of length {chunk_length} added to training data: {num_full_chunks}')
                        
                    
                    # Training data: silences included
                    if create_training_data and chopped_recording_dir:
                        
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

                            print(f'Out of distribution block. Number of chopped chunks w/ intertrial silences of length {chunk_length} added to training data: {num_full_chunks}')
            
                    if full_recording_dir:
                        file_name = f'{full_recording_dir}/{file}.wav'
                        sf.write(file_name, nwbfile_electrodes, 16000, subtype='FLOAT')

                        print('Full recording saved as a WAVE file.')

                    if (ecog_tfrecords_dir and 
                        ((self.patient in ('EFC402', 'EFC403') and (block in self.blocks_ID_demo2) or
                         (self.patient in ('EFC400', 'EFC401') and (block in self.blocks_ID_mocha))))):
                        
                        # Create TFRecords for the ECoG data

                        high_gamma = downsample(nwbfile_electrodes, 
                                                self.nwb_sr, 
                                                self.target_sr, 
                                                'NWB',
                                                ZSCORE=True)

                        phoneme_transcriptions = nwbfile.processing['behavior'].data_interfaces['BehavioralEpochs'].interval_series #['phoneme transcription'].timestamps[:]

                        token_type = 'word_sequence'

                        max_seconds_dict = {
                            'phoneme': 0.2,
                            'word': 1.0,
                            'word_sequence': 6.25,
                            'word_piece_sequence': 6.25,
                            'phoneme_sequence': 6.25,
                            'trial': 6.25
                        }

                        if 'phoneme transcription' in phoneme_transcriptions:
                            print(f'Phoneme transcription for block {block} exists.')
                            phoneme_transcript = phoneme_transcriptions['phoneme transcription']
                            phoneme_onset_times = phoneme_transcript.timestamps[
                                phoneme_transcript.data[()] == 1]
                            phoneme_offset_times = phoneme_transcript.timestamps[
                                phoneme_transcript.data[()] == -1]
                        else:
                            phoneme_transcript = None
                            phoneme_onset_times = None
                            phoneme_offset_times = None

                        example_dicts = []

                        for index, trial in enumerate(nwbfile.trials):
                            t0 = float(trial.iloc[0].start_time)
                            tF = float(trial.iloc[0].stop_time)
                        
                            i0 = np.rint(self.target_sr * t0).astype(int)
                            iF = np.rint(self.target_sr * tF).astype(int)
                            
                            # ECOG (C) SEQUENCE
                            c = high_gamma[i0:iF,:]
                            # print(c.shape)
                            # plt.plot(c[:,0])
                            # break
                        
                            nsamples = c.shape[0]
                            
                            # TEXT SEQUENCE
                            speech_string = trial['transcription'].values[0]
                            text_sequence = sentence_tokenize(speech_string.split(' ')) # , 'text_sequence')
                            
                            # AUDIO SEQUENCE    
                            audio_sequence = []
                            
                            # PHONEME SEQUENCE
                            
                            M = iF - i0
                            
                            max_seconds = max_seconds_dict.get(token_type) # , 0.2) # i don't think this 0.2 default is necessary for the scope of this
                            max_samples = int(np.floor(self.target_sr * max_seconds))
                            max_length = min(M, max_samples)
                            
                            phoneme_array = transcription_to_array(
                                            t0, tF, phoneme_onset_times, phoneme_offset_times,
                                            phoneme_transcript, max_length, self.target_sr 
                                        )
                            
                            phoneme_sequence = [ph.encode('utf-8') for ph in phoneme_array]
                            
                            if len(phoneme_sequence) != nsamples:
                                if len(phoneme_sequence) > nsamples:
                                    phoneme_sequence = [phoneme_sequence[i] for i in range(nsamples)]
                                else:
                                    for i in range(nsamples - len(phoneme_sequence)):
                                        phoneme_sequence.append(phoneme_sequence[len(phoneme_sequence) - 1])
                            
                            print('\n------------------------')
                            print(f'For sentence {index}: ')
                            print(c[0:5,0:5])
                            print(f'Latent representation shape: {c.shape} (should be [samples, nchannel])')
                            print(text_sequence)
                            print(f'Audio sequence: {audio_sequence}')
                            print(f'Length of phoneme sequence: {len(phoneme_sequence)}')
                            print(phoneme_sequence)
                            print('------------------------\n')
                            
                            example_dicts.append({'ecog_sequence': c, 'text_sequence': text_sequence, 'audio_sequence': [], 'phoneme_sequence': phoneme_sequence,})

                        # all_example_dict.extend(example_dicts)
                        # print(len(example_dicts))
                        # print(len(all_example_dict))
                        write_to_Protobuf(f'{ecog_tfrecords_dir}/{self.patient}_B{block}.tfrecord', example_dicts) 

                        print('In distribution block. TFRecords created.')

            except Exception as e: 
                print(f"An error occured and block {path} is not inluded in the wav2vec training data: {e}")

            io.close()


'''
JGM is the author of the following functions. Light modifications made.
'''

def sentence_tokenize(token_list): # token_type = word_sequence
    tokenized_sentence = [
                (token.lower() + '_').encode('utf-8') for token in token_list
            ]
    return tokenized_sentence

def write_to_Protobuf(path, example_dicts):
    '''
    Collect the relevant ECoG data and then write to disk as a (google)
        protocol buffer.
    '''
    writer = tf.io.TFRecordWriter(
        path)
    for example_dict in example_dicts:
        feature_example = tfh.make_feature_example(example_dict)
        writer.write(feature_example.SerializeToString())

def transcription_to_array(trial_t0, trial_tF, onset_times, offset_times, transcription, max_length, sampling_rate):
    
    # if the transcription is missing (e.g. for covert trials)
    if transcription is None:
        return np.full(max_length, 'pau', dtype='<U5')

    # get just the parts of transcript relevant to this trial
    trial_inds = (onset_times >= trial_t0) * (offset_times < trial_tF)
    transcript = np.array(transcription.description.split(' '))[trial_inds]
    onset_times = np.array(onset_times[trial_inds])
    offset_times = np.array(offset_times[trial_inds])

    # vectorized indexing
    sample_times = trial_t0 + np.arange(max_length)/sampling_rate
    indices = (
        (sample_times[None, :] >= onset_times[:, None]) *
        (sample_times[None, :] < offset_times[:, None])
    )

    # no more than one phoneme should be on at once...
    try:
        # print('exactly one phoneme:', np.all(np.sum(indices, 0) == 1))
        assert np.all(np.sum(indices, 0) < 2)
    except:
        pdb.set_trace()

    # ...but there can be locations with *zero* phonemes; assume 'pau' here
    transcript = np.insert(transcript, 0, 'pau')
    indices = np.sum(indices*(np.arange(1, len(transcript))[:, None]), 0)

    return transcript[indices]