import torch
import torch.nn as nn

from .UniLSTM import UniLSTM
from .SimpleBiLSTM import SimpleBiLSTM
from .Baseline import Baseline
from .BiLSTM import BiLSTM

class NLINet(nn.Module):

    def __init__(self, vocab_size, hidden_dim, vocab_vectors, output_dim, encoder_flag, fc_dim, num_layers=1):
        super(NLINet, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])
        # nn.init.xavier_uniform_(self.embedding.weight) # 初始化权重
        self.embedding.weight.data.copy_(vocab_vectors) # 载入预训练词向量
        self.embedding.weight.requires_grad = False

        # encoder layer
        if encoder_flag == "Baseline":
            self.encoder_model = Baseline()
            classifier_input_dim = vocab_size[1]*4
        elif encoder_flag == "UniLSTM":
            self.encoder_model = UniLSTM(vocab_size, hidden_dim)
            classifier_input_dim = hidden_dim*4
        elif encoder_flag == "SimpleBiLSTM":
            self.encoder_model = SimpleBiLSTM(vocab_size, hidden_dim)
            classifier_input_dim = hidden_dim*4*2
        elif encoder_flag == "BiLSTM":
            self.encoder_model = BiLSTM(vocab_size, hidden_dim)
            classifier_input_dim = hidden_dim*4*2
        else:
            raise Exception("encoder_flag should be one of ['UniLSTM',\
                             'SimpleBiLSTM', 'Baseline','BiLSTM' ]")

        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fc_dim),
            nn.Linear(fc_dim, fc_dim),
            nn.Linear(fc_dim, output_dim)
        )


    def forward(self, hypothesis, premise):

        hypothesis, hypothesis_len = hypothesis
        premise, premise_len = premise

        embedding_hypothesis = self.embedding(hypothesis)
        embedding_premise = self.embedding(premise)

        hidden_premise = self.encoder_model(embedding_premise,premise_len)
        hidden_hypothesis = self.encoder_model(embedding_hypothesis,hypothesis_len)

        # classifier
        features = torch.cat((hidden_premise, hidden_hypothesis,
                              torch.abs(hidden_premise-hidden_hypothesis),
                              hidden_premise*hidden_hypothesis), 1)
        output = self.classifier(features)

        return output
