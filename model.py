# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-11-20 15:24:59
'''
import argparse
import pickle

import numpy as np
import yaml
from pyhanlp import HanLP

import torch
import torch.nn.functional as F
import torch.optim as optim
from data_manager import DataManager
from torch import nn
from torch.autograd import Variable

PARSER = argparse.ArgumentParser()

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.W_1 = nn.Parameter(
            torch.randn(self.hidden_size, 1)
        )
        self.b_1 = nn.Parameter(
            torch.randn(1)
        )
    
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden = hidden_states.view(-1, self.hidden_size)
        features = torch.matmul(hidden, self.W_1) + self.b_1
        features_projection = torch.tanh(features)
        u = features_projection.view(batch_size, 1, -1)
        
        alpha = F.softmax(u, -1)
        d = torch.matmul(alpha, hidden_states)
        return d

class RelationshipClassifierModel(nn.Module):
    
    def __init__(self, 
            embedding_size=100,
            vocab_size=20,
            num_label=4,
            hidden_size=128,
            dropout=1,
            batch_size=1
        ):
        super(RelationshipClassifierModel, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_label = num_label
        self.dropout = dropout
        self.batch_size = batch_size
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.attention = Attention(self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_label)

     
    def init_hidden(self):
        return torch.randn(2, self.batch_size, self.hidden_size)

    def forward(self, vec, length):
        self.batch_size = vec.shape[0]
        self.hidden = self.init_hidden()
        embeddings = self.word_embeddings(vec).contiguous()

        _, index_sort = torch.sort(length, dim=0, descending=True)
        _, index_unsort = torch.sort(index_sort, dim=0)

        sorted_embeddings = embeddings.index_select(0, index_sort)
        sorted_length = list(length[index_sort])
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(sorted_embeddings, sorted_length, batch_first=True)
        gru_out, self.hidden = self.gru(packed_embeddings, self.hidden)
        gru_out_padded = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        gru_logits = gru_out_padded[0].index_select(0, index_unsort)
        gru_logits = gru_logits.contiguous().view(self.batch_size, -1, self.hidden_size)

        attention_features = self.attention(gru_logits)
        logits = self.linear(attention_features).contiguous().view(self.batch_size, self.num_label)
        return logits

class RelationshipClassifier(object):
    
    def __init__(self, pre_trained=False, entry="train"):
        self.pre_trained = pre_trained

        self.load_config()
        self.__init_model(entry)

    def __init_model(self, entry):
        if entry == "train":
            if self.pre_trained:
                self.vocab, embeddings = self.load_word_embeddings()
                self.embedding_size = embeddings.shape[-1]
                self.vocab_size = len(self.vocab.keys())
            else:
                params = self.load_params()
                self.embedding_size = params.get("embedding_size")
                self.vocab_size = params.get("vocab_size")
                self.vocab = params.get("vocab")
            self.train_manager = DataManager(batch_size=self.batch_size, vocab=self.vocab)
            self.label_dict = self.train_manager.label_dict

            self.model = RelationshipClassifierModel(
                embedding_size=self.embedding_size,
                vocab_size=self.vocab_size,
                num_label=len(self.label_dict.keys()),
                dropout=0.5
            )
            if self.pre_trained:
                self.model.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))

            self.total_size = len(self.train_manager.batch_data)
            self.vocab_size = len(self.train_manager.vocab)
            
            params = {
                "embedding_size":self.embedding_size,
                "batch_size":self.train_manager.batch_size,
                "vocab_size":self.train_manager.vocab_size,
                "vocab":self.train_manager.vocab,
                "label":self.label_dict
            }
            self.save_params(params)
            self.restore_model()
        elif entry == "predict":
            params = self.load_params()
            self.embedding_size = params.get("embedding_size")
            self.vocab_size = params.get("vocab_size")
            self.vocab = params.get("vocab")
            self.label_dict = params.get("label")
            self.model = RelationshipClassifierModel(
                embedding_size=self.embedding_size,
                vocab_size=self.vocab_size,
                num_label=len(self.label_dict.keys()),
                dropout=1
            )
            self.restore_model()
    
    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 100,
                "hidden_size": 128,
                "batch_size": 2,
                "dropout":0.5,
                "model_path": "models/",
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def load_word_embeddings(self):
        print("load pretrained word embeddings")
        vocab = {}
        embeddings = []
        with open("data/sgns.sogou.bigram") as fopen:
        # with open("data/embeddings") as fopen:
            index = 0
            for line in fopen:
                if index == 0:
                    index += 1
                    continue
                line = line.strip()
                word = line.split()[0]
                embedding = line.split()[1:]
                if word not in vocab:
                    vocab[word] = len(vocab.keys())
                    embeddings.append(embedding)
        self.embedding_size = len(embeddings[-1])
        
        embeddings.append(np.random.randn(self.embedding_size))
        vocab["__UNK__"] = len(vocab.keys())

        embeddings.append(np.random.randn(
            self.embedding_size))
        vocab["__ENTITY__"] = len(vocab.keys())

        embeddings = np.array(embeddings, dtype=np.float32)
        print("pretrained vocab size {}".format(len(vocab.keys())))
        print("pretrained embedding shape {}".format(embeddings.shape))
        return vocab, embeddings

    def train(self):
        criterion= nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(100):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()
                vecs, labels, length = zip(*batch)

                vecs_tensor = torch.tensor(vecs, dtype=torch.long)
                lables_tensor = torch.tensor(labels, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                outputs = self.model(vecs_tensor, length_tensor)
                loss = criterion(outputs, lables_tensor)
                progress = ("█"*int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                        epoch, progress, index, self.total_size, loss.cpu().tolist()
                    )
                )
                print("-"*50)
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_path+'params.pkl')
            
    def predict(self, input_str):
        self.unk = self.vocab.get("__UNK__")
        segments = [i.word for i in HanLP.segment(input_str)]
        vec = [self.vocab.get(i, self.unk) for i in segments]

        input_tensor = torch.tensor(vec, dtype=torch.long).unsqueeze(0)
        length = torch.tensor([len(vec)], dtype=torch.long)
        logits = self.model(input_tensor, length)
        print(torch.max(logits, -1)[1])

if __name__ == "__main__":
    PARSER.add_argument("-pt", "--pre_trained", type=bool, default=False, help="是否使用预训练词向量")
    PARSER.add_argument("-e", "--entry", type=str, default="train", help="入口:train/predict")
    PARSER.add_argument("-i", "--input_str", type=str, default="1月19日，日前__ENTITY__宣布与__ENTITY__达成合作", help="待预测的句子")
    ARGS = PARSER.parse_args()
    rc = RelationshipClassifier(pre_trained=ARGS.pre_trained, entry=ARGS.entry)
    if ARGS.entry == "train":
        rc.train()
    elif ARGS.entry == "predict":
        rc.predict(ARGS.input_str)
