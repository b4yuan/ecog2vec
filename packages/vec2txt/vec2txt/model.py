import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=10):
        super(DynamicRNN, self).__init__()
        
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        output_size = int(output_size)
        
        self.conv1 = nn.Conv1d(input_size, input_size, 5, stride=3)
        self.conv2 = nn.Conv1d(input_size, input_size, 3, stride=2)
        self.conv3 = nn.Conv1d(input_size, input_size, 3, stride=2)
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, sequence_lengths):

        x = x.reshape(x.shape[2],-1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.reshape(1, x.shape[1], -1)
        
        packed_input = pack_padded_sequence(x, sequence_lengths, batch_first=True, enforce_sorted=False)
        
        print(x.shape, packed_input.data.shape, sequence_lengths)
        h_0 = self.init_hidden(x.size(0))
        
        packed_output, _ = self.rnn(packed_input, h_0)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        out = self.fc(output)
        
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()