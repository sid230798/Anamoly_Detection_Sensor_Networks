import torch
import torch.nn as nn

class model(nn.Module) :

    def __init__(self, input, hidden_units, seq_len, pred_len) :

        super(model, self).__init__()
        ## Batch First ensures input to be of shape (batch, seq, input)
        self.lstm1 = nn.LSTM(input, hidden_units, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_units, hidden_units, num_layers=1, batch_first=True)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=hidden_units*seq_len, out_features=pred_len)
        self.dense2 = nn.Linear(in_features=hidden_units*seq_len, out_features=pred_len)

        self.input = input
        self.hidden_units = hidden_units
        #init_hidden()

    def init_hidden(self, batch_size):

        self.batch_size = batch_size
        self.hidden_state1 = (torch.zeros(1, self.batch_size, self.hidden_units), torch.zeros(1, self.batch_size, self.hidden_units))
        self.hidden_state2 = (torch.zeros(1, self.batch_size, self.hidden_units), torch.zeros(1, self.batch_size, self.hidden_units))

    def forward(self, x) :

        ## x of shape (batch, seq, input)
        out1, self.hidden_state1 = self.lstm1(x, self.hidden_state1)
        out1 = self.relu(out1)
        out2, self.hidden_state2 = self.lstm2(out1, self.hidden_state2)
        out2 = self.relu(out2)
        #print(out2.shape, self.batch_size)
        temp = self.dense1(out2.reshape(self.batch_size, -1))
        humid = self.dense2(out2.reshape(self.batch_size, -1))

        #temp = temp.unsqueeze(dim=2)
        #humid = humid.unsqueeze(dim=2)

        return temp, humid