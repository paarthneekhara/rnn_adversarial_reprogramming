import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

USE_CUDA = False

class seq_rewriter(nn.Module):
    def __init__(self, options):
        super(seq_rewriter, self).__init__()
        self.options = options
        
        self.char_embedding = nn.Embedding(options['vocab_size'], options['vocab_size'])
        self.char_embedding.weight.data = torch.eye(options['vocab_size'])
        self.char_embedding.weight.requires_grad = False

        self.conv1 = nn.Conv1d(options['vocab_size'], options['target_size'], kernel_size = 3, padding = 1)
        self.saved_log_probs = []
        # self.lstm = nn.LSTM(options['vocab_size'], options['hidden_size'], batch_first = True)
        # self.output_layer = nn.Linear(options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch):

        one_hot = self.char_embedding(sentence_batch)
        logits = self.conv1(one_hot.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        logits = logits.view(logits.size(0) * logits.size(1), logits.size(2))
        probs = F.softmax(logits)

        # prediction = []
        new_seq = torch.zeros(sentence_batch.size(0)*sentence_batch.size(1)).long()
        for i in xrange(probs.size(0)):
            m = Categorical(probs[i])
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            if i % sentence_batch.size(1) < 10:
                new_seq[i] = action.data[0]

            # if new_seq[i] %  

        new_seq = new_seq.view(sentence_batch.size(0), sentence_batch.size(1))
        if USE_CUDA:
            new_seq = new_seq.cuda()
        return new_seq

def main():
    rnn_options = {
        'vocab_size' : 100,
        'hidden_size' : 200,
        'target_size' : 3
    }

    chrrnn = seq_rewriter(rnn_options)
    sent_batch = torch.LongTensor(32, 10).random_(0, 10)
    chrrnn(sent_batch)


if __name__ == '__main__':
    main()