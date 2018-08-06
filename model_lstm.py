import torch.nn as nn
import torch
# from torch.autograd import Variable

class charRNN(nn.Module):
    def __init__(self, options):
        super(charRNN, self).__init__()
        self.options = options
        
        self.char_embedding = nn.Embedding(options['vocab_size'], options['vocab_size'])
        self.char_embedding.weight.data = torch.eye(options['vocab_size'])
        self.char_embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(options['vocab_size'], options['hidden_size'], batch_first = True)
        self.output_layer = nn.Linear(options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch, hidden = None):
        # sentence_batch = Variable(sentence_batch)

        one_hot = self.char_embedding(sentence_batch)
        if not hidden:
            lstm_out, new_hidden = self.lstm(one_hot)
        else:
            lstm_out, new_hidden = self.lstm(one_hot, hidden)
        
        # print lstm_out.shape, lstm_out.shape[0] * lstm_out.shape[1]
        lstm_out = lstm_out.contiguous()
        # print type(lstm_out)
        # print lstm_out.shape
        lstm_out = lstm_out[:,-1,:]
        # lstm_out = lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        logits = self.output_layer(lstm_out)

        return logits

def main():
    rnn_options = {
        'vocab_size' : 100,
        'hidden_size' : 200,
        'target_size' : 2
    }

    chrrnn = charRNN(rnn_options)
    sent_batch = torch.LongTensor(32, 10).random_(0, 10)
    chrrnn(sent_batch)


if __name__ == '__main__':
    main()