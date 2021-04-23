import torch.nn as nn
import torch


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_dim, num_layers=1):
        super(BiLSTM, self).__init__()
        # encoder layer
        self.lstm = nn.LSTM(input_size=vocab_size[1], hidden_size=hidden_dim,
                            num_layers=num_layers, bidirectional=True,
                            batch_first=True)

    def forward(self, x, x_len):
        # sort
        x_len_sorted, idx_ori = torch.sort(x_len,descending=True)
        x_sorted = x.index_select(0,idx_ori)

        x_sorted_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted.cpu(),batch_first=True)

        output_sorted, (hidden_sorted, cell_sorted) = self.lstm(x_sorted_packed)

        output_sorted, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(output_sorted,batch_first=True)

        # recover order
        _, idx2 = torch.sort(idx_ori)
        output_recover = output_sorted.index_select(0,idx2)
        x_len_recover = x_len_sorted.index_select(0,idx2)

        max_pool = []
        for sample, length in zip(output_recover,x_len_recover):
            s_c = sample[0:length]
            max_v = torch.max(s_c,0)
            max_pool.append(max_v.values)

        max_pool = torch.stack(max_pool,dim=0)

        return max_pool

