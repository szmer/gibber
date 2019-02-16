import torch
from torch import nn

class GibberishEstimator(nn.Module):
    
    def __init__(self, embedding, lstm_hidden_size, lstm_layers_count, lstm_is_bidirectional, use_cuda=False):

        super(GibberishEstimator, self).__init__()

        self.lstm_layers_n = lstm_layers_count
        self.lstm_hidden_size = lstm_hidden_size

        self.word_embedding = embedding
        self.estim_layer = nn.LSTM(input_size=embedding.embedding_dim,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_layers_n,
                                bidirectional=lstm_is_bidirectional,
                                batch_first=True)
        self.lin = nn.Linear(self.lstm_hidden_size, 1)
        self.sigmoid_nonlin = nn.Sigmoid()
        self.decider = nn.Sequential(self.lin, self.sigmoid_nonlin)

        self.use_cuda = use_cuda

    def forward(self, words_windows):
        if self.use_cuda:
             words_windows = words_windows.cuda()
        words_repr = self.word_embedding(words_windows)
        cells, hidden = self.estim_layer(words_repr)
        estimations = self.decider(cells)
        estimations = estimations[:, estimations.size(1)-1, :] # leave only the last decision
        return estimations
