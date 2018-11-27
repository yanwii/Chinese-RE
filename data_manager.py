# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-05-30 14:46:36
'''
import copy
import pickle as cPickle
import torch
from pyhanlp import HanLP

class DataManager():
    def __init__(self, max_length=100, batch_size=20, data_type='train', vocab={}, pre_trained=False):
        self.index = 0
        self.vocab_size = 0
        self.pre_trained = pre_trained
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        self.vocab = vocab
        self.new_model = False
        self.label_dict = {}
        # self.tag_map = {"O":0, "B-ORG":1, "I-ORG":2, "E-ORG":3, "B-PER":4, "I-PER":5, "E-PER":6, "S":7}
        if data_type == "train":
            self.data_path = "data/train"
        elif data_type == "dev":
            self.data_path = "data/dev"
            self.load_data_map()
        elif data_type == "test":
            self.data_path = "data/test"
            self.load_data_map()
        self.extra_vocab()

        self.load_data()    
        self.prepare_batch()
    
    def extra_vocab(self):
        if not self.vocab:
            self.new_model = True
        for word in ["__UNK__", "__ENTITY__"]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab.keys())
        self.unk = self.vocab["__UNK__"]

    def load_data_map(self):
        with open("models/data.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    def add_to_vocab(self, word):
        if self.data_type == "train" and self.new_model:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def load_data(self):
        # load data
        # add vocab
        # covert to one-hot
        with open(self.data_path) as fopen:
            for line in fopen:
                vec = []
                line = line.strip()
                entity_1, entity_2, label, sentence = line.split("\t")
                # 替换
                sentence = sentence.replace(entity_1, "__ENTITY__")
                sentence = sentence.replace(entity_2, "__ENTITY__")
                segments = [i.word for i in HanLP.segment(sentence)]

                if label not in self.label_dict:
                    self.label_dict[label] = len(self.label_dict.keys())
                for word in segments:
                    self.add_to_vocab(word)
                    vec.append(self.vocab.get(word, self.unk))
                self.data.append([vec, self.label_dict.get(label, 0)])
        self.vocab_size = len(self.vocab.keys())
        print("{} data: {}".format(self.data_type ,len(self.data)))
        print("vocab size: {}".format(self.vocab_size))
        print("-"*50)
    
    def prepare_batch(self):
        '''
            prepare data for batch
        '''
        index = 0
        while True:
            if index+self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)
    
    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])
        for i in c_data:
            i.append(len(i[0]))
            i[0] = i[0] + (max_length-len(i[0])) * [0]
            # i[0] = torch.tensor(i[0])
            # i[1] = torch.tensor(i[1])
        return c_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
