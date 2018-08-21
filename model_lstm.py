import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch.autograd import Variable

class charRNN(nn.Module):
    def __init__(self, options):
        super(charRNN, self).__init__()
        self.options = options
        print options
        self.char_embedding = nn.Embedding(options['vocab_size'], options['embedding_size'])
        self.lstm = nn.LSTM(options['embedding_size'], options['hidden_size'], batch_first = True)
        self.output_layer = nn.Linear(options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch, hidden = None):
        # sentence_batch = Variable(sentence_batch)

        char_embedding = self.char_embedding(sentence_batch)
        char_embedding = F.tanh(char_embedding)

        if not hidden:
            lstm_out, new_hidden = self.lstm(char_embedding)
        else:
            lstm_out, new_hidden = self.lstm(char_embedding, hidden)
        
        # print lstm_out.shape, lstm_out.shape[0] * lstm_out.shape[1]
        lstm_out = lstm_out.contiguous()
        # print type(lstm_out)
        # print lstm_out.shape
        lstm_out = lstm_out[:,-1,:]
        # lstm_out = lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        logits = self.output_layer(lstm_out)

        return logits


class biRNN(nn.Module):
    def __init__(self, options):
        super(biRNN, self).__init__()
        self.options = options
        print options
        self.drop = nn.Dropout(0.7)
        self.char_embedding = nn.Embedding(options['vocab_size'], options['embedding_size'])
        self.lstm = nn.LSTM(options['embedding_size'], options['hidden_size'], batch_first = True, bidirectional=True)
        self.output_layer = nn.Linear(2*options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch, hidden = None):
        # sentence_batch = Variable(sentence_batch)

        char_embedding = self.char_embedding(sentence_batch)
        char_embedding = F.tanh(char_embedding)
        
        if not hidden:
            lstm_out, new_hidden = self.lstm(char_embedding)
        else:
            lstm_out, new_hidden = self.lstm(char_embedding, hidden)
        
        # print lstm_out.shape, lstm_out.shape[0] * lstm_out.shape[1]
        lstm_out = lstm_out.contiguous()
        # print lstm_out.size()
        # print type(lstm_out)
        # print lstm_out.shape
        lstm_out = lstm_out[:,-1,:] + lstm_out[:,0,:]
        # lstm_out = lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        logits = self.output_layer(lstm_out)

        return logits


class CnnTextClassifier(nn.Module):
    def __init__(self, options, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()
        self.options = options

        self.embedding = nn.Embedding(options['vocab_size'], options['embedding_size'])

        self.convs = nn.ModuleList([
            nn.Conv2d(1, options['hidden_size'], [window_size, options['embedding_size']], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(options['hidden_size'] * len(window_sizes), options['target_size'])

    def forward(self, x):
        x = self.embedding(x)           # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]

        return logits


def main():
    rnn_options = {
        'vocab_size' : 100,
        'hidden_size' : 200,
        'target_size' : 2,
        'embedding_size' : 200,
    }

    chrrnn = CnnTextClassifier(rnn_options)
    sent_batch = torch.LongTensor(32, 10).random_(0, 10)
    # print chrrnn(sent_batch)


if __name__ == '__main__':
    main()