import torch.nn as nn
import torch


class UniLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_dim, num_layers=1):
        super(UniLSTM, self).__init__()
        # encoder layer
        self.lstm = nn.LSTM(input_size=vocab_size[1], hidden_size=hidden_dim,
                            num_layers=num_layers, bidirectional=False,
                            batch_first=True)

    def forward(self, x, x_len):
        # sort
        x_len_sorted, idx_ori = torch.sort(x_len,descending=True)
        x_sorted = x.index_select(0,idx_ori)

        x_sorted_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted.cpu(),batch_first=True)

        output_sorted, (hidden_sorted, cell_sorted) = self.lstm(x_sorted_packed)

        hidden_sorted = hidden_sorted.view(hidden_sorted.size()[1], -1)
        # hidden = torch.cat([hidden[-4], hidden[-3], hidden[-2], hidden[-1]], dim=1) # 双向时、多层时

        # recover order
        _, idx2 = torch.sort(idx_ori)
        hidden_recover = hidden_sorted.index_select(0,idx2)

        return hidden_recover

