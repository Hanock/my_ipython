from __future__ import print_function
from sklearn.preprocessing import normalize
import konlpy
import nltk
import numpy as np
import re 


def create_dic(chat_fname, dic_fname):
    fp_in = open('cafechat.txt', 'rb')
    fp_out = open('cafe_dic.txt', 'wb')

    dic = set()

    for line in fp_in.readlines():
        line = line.decode('utf-8')
        words = konlpy.tag.Twitter().pos(line)
        for pair in words:
            if pair[1] == 'Verb' or pair[1] == 'Noun':
                dic.add(pair[0].encode('utf-8'))

    for word in dic:
        print(word, file=fp_out)

    fp_out.close()
    fp_in.close()

    
class CafeDictionary:
    
    def __init__(self, dic_fname):
        fp = open(dic_fname)
        self.dictionary = dict()
        idx = 0

        for word in fp.readlines():
            word = word.replace('\n', '')
            if len(word) == 0:
                continue
            self.dictionary[word] = idx
            idx = idx + 1

        fp.close()
    
    def get_index(self, word):
        try:
            return self.dictionary[word]
        except KeyError as e:
            return -1
        
    def size(self):
        return len(self.dictionary)
    
    def bow(self, sentence):
        vec = np.zeros([self.size()])
        words = konlpy.tag.Twitter().pos(sentence.decode('utf-8'))
        for pair in words:
            if not self.valid_pumsa(pair[1]):
                continue
                
            idx = self.get_index(pair[0].encode('utf-8'))
            if idx == -1:
                continue
            
            vec[idx] += 1
            
        return vec
        
    def valid_pumsa(self, pumsa):
        return pumsa == 'Verb' or pumsa == 'Noun'
    
    def sentence_similarity(self, sentence1, sentence2):
        v1 = self.bow(sentence1)
        v2 = self.bow(sentence2)
        
        return self.bow_similarity(v1, v2)
    
    def bow_similarity(self, v1, v2):
        return np.dot(normalize(v1), normalize(v2).T)
        
        
        
class CafeChat:
    
    def __init__(self, dic_fname, chat_fname):
        self.dictionary = CafeDictionary(dic_fname)
        self.parse_chat_file(chat_fname)
        
    def parse_chat_file(self, chat_fname):
        fp = open(chat_fname)
        lines = fp.readlines()
        
        self.person = []
        self.chat = []
        self.bow = np.zeros([len(lines), self.dictionary.size()])
        self.chat_size = len(lines)
        
        i = 0
        for line in lines:
            parts = line.split("\t")
            self.person.append(parts[0])
            self.chat.append(parts[1])
            self.bow[i] = self.dictionary.bow(parts[1])
            i += 1

        fp.close()
    
    def get_response(self, sentence):
        max_idx = -1
        max_sim = 0
        bow = self.dictionary.bow(sentence)
    
        for i in np.arange(self.chat_size):
            p = self.person[i]
            if p == 'c':
                continue
            sim = self.dictionary.bow_similarity(bow, self.bow[i])
            if max_sim < sim:
                max_idx = i
                max_sim = sim
    
        if max_idx == -1:
            return -1
        if max_idx + 1 >= self.chat_size:
            return -2
        
        return self.chat[max_idx+1]
