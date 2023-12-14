
import os
import pdb

from pynwb import NWBHDF5IO
import numpy as np
import os
import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchsummary import summary
from vec2txt.model import DynamicRNN
from vec2txt.utils import word_error_rate

# Creating a list of dictionaries of:
# 
# `ecog_sequence`: ECoG data, clipped to token(-sequence) length
# `text_sequence`: the corresponding text token(-sequence)
# `audio_sequence`: the corresponding audio (MFCC) token sequence (gonna set to)
# `phoneme_sequence`: ditto for phonemes--with repeats
#

def transcription_to_array(trial_t0, trial_tF, onset_times, offset_times, transcription, max_length, sampling_rate):
    # Borrowed from jgmakin/ecog2txt
    
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

def sentence_tokenize(token_list): # token_type = word_sequence
    # Borrowed from jgmakin/ecog2txt
    
    tokenized_sentence = [
                (token.lower() + '_').encode('utf-8') for token in token_list
            ]
    return tokenized_sentence
            
# sorting function for latent representation filenames
def custom_sort_key(filename):
    num_part = int(filename.split('nwb_')[1].split('.wav.pt')[0])
    return num_part

def get_example_dicts(subject, blocks, ecog_representation_path):
    # Modified from jgmakin/ecog2txt
    
    all_example_dict = []
    for block in blocks:

        nwb_filepath = folder_path = f"/NWB/EFC{subject}/EFC{subject}_B{block}.nwb"
        io = NWBHDF5IO(nwb_filepath, load_namespaces=True, mode='r')
        nwbfile = io.read()
        
        ### GETTING LATENT REPRESENTATION PATHS ###
        c_vectors_dir = ecog_representation_path
        prefix = f'EFC{subject}_B{block}.nwb_' # EFC400_B72.nwb_49.wav.pt

        c_file_path_list = []

        for filename in os.listdir(c_vectors_dir):
            if filename.startswith(prefix):
                c_file_path_list.append(os.path.join(c_vectors_dir, filename))
                
        c_file_path_list = sorted(c_file_path_list, key=custom_sort_key)
        ###########################################
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

        makin_sr = 200 # screw it, actually just pass in c_sr everywhere
        c_sr = 101# 399# 18

        for index, trial in enumerate(nwbfile.trials):
            
            # ECOG (C) SEQUENCE
            c_filepath = c_file_path_list[index]
            # print(c_filepath)
            c = torch.load(c_filepath)#.detach().numpy() # [1, nchannel, samples]
            # c = c.reshape(512,-1) # [nchannel, samples]
            c = c.T # [samples, nchannel]
            
            nsamples = c.shape[0]
            
            # TEXT SEQUENCE
            speech_string = trial['transcription'].values[0]
            text_sequence = sentence_tokenize(speech_string.split(' ')) # , 'text_sequence')
            
            # AUDIO SEQUENCE    
            audio_sequence = []
            
            # PHONEME SEQUENCE
            t0 = float(trial.iloc[0].start_time)
            tF = float(trial.iloc[0].stop_time)
        
            i0 = np.rint(c_sr*t0).astype(int)
            iF = np.rint(c_sr*tF).astype(int)
            
            M = iF - i0
            
            max_seconds = max_seconds_dict.get(token_type) # , 0.2) # i don't think this 0.2 default is necessary for the scope of this
            max_samples = int(np.floor(c_sr*max_seconds))
            max_length = min(M, max_samples)
            
            phoneme_array = transcription_to_array(
                            t0, tF, phoneme_onset_times, phoneme_offset_times,
                            phoneme_transcript, max_length, c_sr # makin_sr
                        )
            
            phoneme_sequence = [ph.encode('utf-8') for ph in phoneme_array]
            
            if len(phoneme_sequence) != nsamples:
                if len(phoneme_sequence) > nsamples:
                    phoneme_sequence = [phoneme_sequence[i] for i in range(nsamples)]
                else:
                    for i in range(nsamples - len(phoneme_sequence)):
                        phoneme_sequence.append(phoneme_sequence[len(phoneme_sequence) - 1])
            
            # print('\n------------------------')
            # print(f'For sentence {index}: ')
            # print(c[0:5,0:5])
            # print(f'Latent representation shape: {c.shape} (should be [samples, nchannel])')
            # print(text_sequence)
            # print(f'Audio sequence: {audio_sequence}')
            # print(f'Length of phoneme sequence: {len(phoneme_sequence)}')
            # print(phoneme_sequence)
            # print('------------------------\n')
            
            example_dicts.append({'ecog_sequence': c, 'text_sequence': text_sequence, 'audio_sequence': [], 'phoneme_sequence': phoneme_sequence,})
            
        all_example_dict.extend(example_dicts)

    return all_example_dict

def get_decoding_inputs_and_targets(all_example_dicts):
    """
    From a dictionary with all examples
    
    Returns a list of unique words;
            the latent representation of all sentences
            the one hot encoded vector for each sentence
    """
    c_vectors = [example['ecog_sequence'] for example in all_example_dicts]
    c_vectors = [torch.tensor(c_vectors[i]).reshape(1,-1,512) for i in range(len(c_vectors))]
    print((c_vectors[0].shape))
    texts = [example['text_sequence'] for example in all_example_dicts]
    print(texts[0])

    unique_words = set()

    for text in texts:
        for word in text:
            word = word.decode('utf-8')
            unique_words.update({word})

    unique_words = sorted(unique_words)
    print(f"Vocabulary size: {len(unique_words)}")
    text_one_hot = []

    for text in texts:
        one_hot = torch.zeros(1, len(text))
        for i, word in enumerate(text):
            word = word.decode('utf-8')
            one_hot[0, i] = unique_words.index(word)
        text_one_hot.append(one_hot)

    c_vectors = [c_vector.reshape(-1, 512) for c_vector in c_vectors]
    text_one_hot = [one_hot.reshape(-1) for one_hot in text_one_hot]
    # unique_words = sorted(list(unique_words))

    print(f"Number of data: {len(all_example_dicts)}")
    print(f"Example sentence: {(text_one_hot[0])}")
    return unique_words, c_vectors, text_one_hot


blocks = [4]#, 41, 57, 61, 66, 69, 73, 77, 83, 87] # [3,4,6,8,10,12,14,15,19,23,28,30,38,40,42,46,57,61,72] # change this for what block you're making
subject = '401'
ecog_representation_path = '/home/bayuan/Documents/fall23/ecog2vec/wav2vec_outputs/raw_hg_entirerecording_sentences'

all_example_dicts = get_example_dicts(subject, blocks, ecog_representation_path)

unique_words, c, t = get_decoding_inputs_and_targets(all_example_dicts)

input_size = c[0].shape[1]
hidden_size = 256
output_size = len(unique_words)
num_epochs = 40
lr = 0.0001

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

rnn = DynamicRNN(int(input_size), int(hidden_size), int(output_size))
rnn = rnn.to(DEVICE)
print(next(rnn.parameters()).device)
summary(rnn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=lr)

total_vectors = len(c)

split_index = int(0.9 * total_vectors)
print(f"Number of training data: {split_index}")

train_c_vectors = c[:split_index]
train_text_one_hot = t[:split_index]
test_c_vectors = c[split_index:]
test_text_one_hot = t[split_index:]

train_data = __builtins__.list(zip(train_c_vectors, train_text_one_hot))
train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_data = __builtins__.list(zip(test_c_vectors, test_text_one_hot))
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)

train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    rnn.train()  # Set the model in training mode

    total_train_loss = 0.0
    for ecog_seq, target_seq in train_data_loader:
        ecog_seq = ecog_seq.to('cuda')
        target_seq = target_seq.to('cuda')
        
        optimizer.zero_grad()
        ecog_seq_padded = pad_sequence(ecog_seq.float(), batch_first=True)
        target_seq_padded = pad_sequence(target_seq, batch_first=True)
        
        sequence_lengths = torch.tensor([target_seq_padded.shape[1]]).cuda()

        output = rnn(ecog_seq_padded, sequence_lengths.to('cpu'))
        
        
        output = output.reshape(-1, output_size)
        target_seq_padded = target_seq_padded.reshape(-1)
        
        output = output.to('cuda')
        target_seq_padded = target_seq_padded.to('cuda')

        loss = criterion(output, target_seq_padded.long())

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
    rnn.eval()
        
    total_valid_loss = 0.0
    
    with torch.no_grad():
        for ecog_seq, target_seq in test_data_loader:
            ecog_seq = ecog_seq.to('cuda')
            target_seq = target_seq.to('cuda')
            
            ecog_seq_padded = pad_sequence(ecog_seq, batch_first=True)
            target_seq_padded = pad_sequence(target_seq, batch_first=True)
            
            sequence_lengths = torch.tensor([target_seq_padded.shape[1]]).cuda()

            output = rnn(ecog_seq_padded.float(), sequence_lengths.to('cpu'))
            output = output.reshape(-1, output_size)
            target_seq_padded = target_seq_padded.reshape(-1)
            
            output = output.to('cuda')
            target_seq_padded = target_seq_padded.to('cuda')
            
            loss = criterion(output, target_seq_padded.long())
            
            total_valid_loss += loss.item()
        
    average_train_loss = total_train_loss / len(train_data_loader)  # Calculate the average loss for the epoch
    average_valid_loss = total_valid_loss / len(test_data_loader)
    train_loss.append(average_train_loss)
    valid_loss.append(average_valid_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_train_loss:.30f}")

print("Training finished.")

print("Beginning eval.")

rnn.eval()

total_loss = 0.0
total_wer = 0

with torch.no_grad():
    
    for ecog_seq, target_seq in test_data_loader:
        ecog_seq = ecog_seq.to('cuda')
        target_seq = target_seq.to('cuda')
        
        predicted_sequence = []
        target_sentence = []
        
        sequence_lengths = torch.tensor([target_seq.shape[1]]).cuda()

        ecog_seq_padded = pad_sequence(ecog_seq.float(), batch_first=True)
        target_seq_padded = pad_sequence(target_seq, batch_first=True)

        output = rnn(ecog_seq_padded, sequence_lengths.to('cpu'))
        
        output = output.to('cuda')
        target_seq_padded = target_seq_padded.to('cuda')

        output = output.reshape(-1, output_size)
        for time_step in range(output.shape[0]):
            predicted_word_index = output[time_step].argmax().item()
            predicted_word = unique_words[predicted_word_index] if predicted_word_index < output_size else 'unknown'  # Use 'unknown' for out-of-vocabulary words
            predicted_sequence.append(predicted_word)
        for time_step in range(len(target_seq[0])):
            tensor = target_seq[0]
            word = unique_words[int(tensor[time_step])]
            target_sentence.append(word)
            
        wer = word_error_rate(target_sentence, predicted_sequence)
        total_wer += wer
        print("Example sentence:")
        print("     Predicted Sequence:", ' '.join(predicted_sequence))
        print("     Target Sequence:", ' '.join(target_sentence))
        print(f"    WER: {wer}")
        target_seq_padded = target_seq_padded.reshape(-1)

        loss = criterion(output, target_seq_padded.long())

        total_loss += loss.item()
        

average_loss = total_loss / len(test_data_loader)

print(f"Evaluation Loss: {average_loss:.4f}")
print(f"Avg test WER: {total_wer / len(test_data_loader)}")

# Visualizing train/valid curves
import matplotlib.pyplot as plt

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Train Loss', linestyle='-')

plt.plot(epochs, valid_loss, label='Validation Loss', linestyle='-')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.show()