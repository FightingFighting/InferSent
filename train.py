# 导入torchtext相关包
import os

from torchtext import data

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchtext.datasets import SNLI
from torchtext.vocab import GloVe

from model.NLINet import NLINet

import logging

from torch.utils.tensorboard import SummaryWriter

def binary_acc(preds, y):
    """
    get accuracy
    """
    preds = torch.argmax(preds, dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum()
    return acc


def main(args):

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    logging.basicConfig(filename=os.path.join(args.output_dir,"output.log"),
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    writer = SummaryWriter(os.path.join(args.output_dir,"tensorboard"))


    # build dataset and word embedding
    glove = GloVe(name='840B', dim=args.embedding_dim, cache="./dataset/.vector_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #set up fields
    text_field = data.Field(tokenize='spacy',tokenizer_language="en_core_web_sm",lower=True,include_lengths=True,batch_first=True)
    label_field = data.Field(sequential=False)

    train, val, test = SNLI.splits(text_field, label_field, root="./dataset/.data")

    # build vocab
    text_field.build_vocab(train, vectors=glove)
    label_field.build_vocab(train)
    vocabulary = text_field.vocab


    train_iters,val_iters,test_iters = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size_train, device=device)

    # build model
    vocab_size = vocabulary.vectors.size()
    NLINet_model = NLINet(vocab_size=vocab_size, hidden_dim=args.hidden_dim_LSTM,
                          vocab_vectors= vocabulary.vectors, output_dim=args.output_dim,
                          encoder_flag=args.encoder_flag, fc_dim = args.fc_dim)
    NLINet_model.to(device)


    # optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, NLINet_model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    best_acc = 0
    save_file_list=[]
    for epoch in range(args.epochs):

        train_acc = 0
        train_loss = 0

        logging.info(f"***********Start epoch:{epoch}/{args.epochs}***********")
        logging.info(f"lr:{optimizer.param_groups[0]['lr']}")
        NLINet_model.train()
        for i, batch in enumerate(train_iters):
            hypothesis= batch.hypothesis
            premise = batch.premise
            labels = batch.label-1

            # forward + backward + optimize
            preds = NLINet_model(hypothesis, premise)
            loss = criterion(preds,labels)

            print(f"train, epoch:{epoch}/{args.epochs},step:{i}/{int(len(train_iters.dataset.examples)/args.batch_size_train)},loss:{loss}")
            acc_temp = binary_acc(preds, labels).item()
            train_acc += acc_temp
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write tensorboard
            if epoch == 0 and i == 0:
                writer.add_graph(NLINet_model,(hypothesis, premise))

            writer.add_scalar('training loss',
                              loss.item() / len(labels),
                              epoch * len(train_iters) + i)
            writer.add_scalar('training accuracy',
                              acc_temp / len(labels),
                              epoch * len(train_iters) + i)


        train_acc = train_acc/len(train_iters.dataset.examples)
        train_loss = train_loss/len(train_iters.dataset.examples)

        # update learning rate
        scheduler.step()

        # val
        NLINet_model.eval()
        val_acc = 0
        val_loss = 0
        for j, batch in enumerate(val_iters):
            hypothesis = batch.hypothesis
            premise = batch.premise
            labels = batch.label-1

            # forward + backward + optimize
            preds = NLINet_model(hypothesis, premise)
            loss = criterion(preds,labels)

            print(f"val, epoch:{epoch}/{args.epochs},step:{j}/{int(len(val_iters.dataset.examples)/args.batch_size_train)},loss:{loss}")
            acc_temp = binary_acc(preds, labels).item()
            val_acc += acc_temp
            val_loss += loss.item()

            # write tensorbooard
            writer.add_scalar('val loss',
                              loss.item() / len(labels),
                              epoch * len(val_iters) + j)
            writer.add_scalar('val accuracy',
                              acc_temp / len(labels),
                              epoch * len(val_iters) + j)


        val_acc = val_acc/len(val_iters.dataset.examples)
        val_loss = val_loss/len(val_iters.dataset.examples)

        logging.info(f"train_loss:{train_loss:.5f}-val_loss:{val_loss:.5f}-train_acc: {train_acc:.5f}-val_acc: {val_acc:.5f}")


        if optimizer.param_groups[0]['lr'] < 0.00001:
            break

        # save
        if best_acc <= val_acc:
            best_acc = val_acc
            if epoch == 0:
                os.makedirs(os.path.join(args.output_dir, "models"))

            torch.save(NLINet_model,
                   os.path.join(args.output_dir, "models",f'epoch{epoch}_checkpoint.pkl'))
            save_file_list.append(f'epoch{epoch}_checkpoint.pkl')

            torch.save(NLINet_model,
                       os.path.join(args.output_dir, "models",f'best_checkpoint.pkl'))

            if len(save_file_list)>5:
                delete_file = save_file_list.pop(0)
                os.remove(os.path.join(args.output_dir, "models", delete_file))
        else:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5.0
            print("lr:",optimizer.param_groups[0]['lr'])

    writer.close()
    print('train completed')
    logging.info("********************train completed***********************")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train',
        description='train model')

    parser.add_argument('--embedding_dim', default=300, type=int,
                        help="the dimension of the vocab's embedding")

    parser.add_argument('--batch_size_train', default=64, type=int,
                        help="the batch size of the train phase")

    parser.add_argument('--batch_size_val', default=64, type=int,
                        help="the batch size of val phase")

    parser.add_argument('--lr', default=0.1, type=float,
                        help="learning rate")

    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help="learning rate weight decay")

    parser.add_argument('--hidden_dim_LSTM', default=2048, type=int,
                        help="the dimension of hidden layer in LSTM")

    parser.add_argument('--output_dim', default=3, type=int,
                        help="the dimension of output layer")

    parser.add_argument('--epochs', default=100, type=int,
                        help="the number of epoch")

    parser.add_argument("--fc_dim", type=int, default=512, help="the dimension of fc layer")


    parser.add_argument('--encoder_flag', default="BiLSTM", type=str,
                        help="indicate which encoder we choose, it can be one of \
                             ['Baseline', 'UniLSTM', 'SimpleBiLSTM', 'BiLSTM' ]")

    parser.add_argument('--output_dir', default="./output/BiLSTM", type=str,
                        help="output path")

    parser.add_argument("--seed", type=int, default=1234, help="seed")

    args = parser.parse_args()

    os.mkdir(args.output_dir)

    main(args)





