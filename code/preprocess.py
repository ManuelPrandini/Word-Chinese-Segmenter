#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Dict
import numpy as np
from tensorflow.keras.utils import to_categorical
import numpy as np


'''
THIS FILE CONTAINS A SET OF METHODS USED TO PREPROCESS THE FILES THAT CONTAIN
THE CHINESE CHARACTERS.
'''

def create_bies_format(line):
    '''
    Method that create the corrispettive bies format of a string
    @param line = the input string
    @return the bias format of the input line
    '''
    i = 0
    bies = ""
    line = line.replace('\n',"")
    line = line.replace('\u3000',' ')
    for index in range(len(line)):
        if(index < len(line)-1):
            if(line[index] == ' '):
                i = 0
            elif(i == 0 and line[index+1] == ' '):
                i = 0
                bies+='S'
            elif(i == 0 and line[index+1] != ' '):
                bies+='B'
                i+=1
            elif(i > 0 and line[index+1] != ' '):
                bies+='I'
                i+=1
            else :
                bies+='E'
                i=0
        else:
            if(i==0):
                bies+='S'
            elif(i>0):
                bies+='S'
    return bies


def create_bies_lines(path):
    '''
    method used to create an array that contains the bies format lines
    of a specific file passed through the path
    @param path = the pathfile of the file to convert
    @return an array with bies lines
    '''
    bies_lines = []
    with open(path,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            bies = create_bies_format(line)
            if (len(line) != 0):
              bies_lines.append(bies)
        f.close()
    return bies_lines



def delete_spaces_file(path):
    '''
    Method used to create an array that contains the lines without spaces
    @param path input file
    @return an array with sentences without space
    '''
    result = []
    with open(path,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace('\u3000'," ")
            line = line.replace(" ","")
            if(len(line) != 0):
              result.append(line)
        f.close()
    return result


def create_file(path,lines):
    '''
    Method used to create a file containing the lines array
    @param path = the path where to save the file
    @param bies_lines = the array containing the bies format sentences
    '''
    with open(path,'w') as f:
        for line in lines:
            f.write(line+'\n')
        f.close()


def read_file(path):
    '''
    Method that read the file(dataset) from the path and return
    an array of elements
    :param path the path of the file to read
    :return an array containing the elements of the file
    '''
    sentences = []
    with open(path,"r",encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            sentences.append(line)
        f.close()
    return sentences


def split_into_ngrams(sentence: str, n : int):
    """
    Split a sentence in array of n-grams
    :param sentence string to split
    :param n the size of the n-grams
    :return an array of ngrams
    """
    ngrams = []
    for i in range(len(sentence)):
        ngram = sentence[i:i+n]
        if(n >= 2 and i == len(sentence)-1): ngram = ngram+"<END>"
        ngrams.append(ngram)
    return ngrams


def make_vocab(sentences,n):
    '''
    Method that create the vocab starting from an array of lines(sentences)
    :param sentences List of sentences used to build the vocab
    :param n the size of the n-grams
    :return vocab Dictionary that has as key the ngram and as a value the index
    '''
    vocab = {"<PAD>":0,"UNK": 1}
    for sentence in sentences:
        bigrams = split_into_ngrams(sentence,n)
        for bigram in bigrams:
            if bigram not in vocab:
                vocab[bigram] = len(vocab)
    return vocab

def make_vocab_label():
    '''
    Method that create the vocab of the label.
    Each key is a letter of the BIES format.
    :return vocab Dictionary that as key the BIAS letter and as a value the categorical conversion
    '''
    vocab = {"B":0,"I":1,"E":2,"S":3}
    return vocab


def word_to_index(sentences,vocab,n):
    '''Method that convert each entry of the ngram vocab into the correspetive value of the key.
        @param sentences array containing the sentences that we want convert
        @param vocab use to map word to index
        @param n indicates the n-grams split
        @return the vector result into the form of np.array
    '''
    result = []
    for sentence in sentences:
        vec_sentence = []
        split = split_into_ngrams(sentence,n)
        for s in split:
          if s not in vocab:
            vec_sentence.append(vocab["UNK"])
          else:
            vec_sentence.append(vocab[s])
        result.append(np.array(vec_sentence))
    return np.array(result)
            


def find_max_length(vector):
    '''
    Find the max length of all sentences in the vector passed in input
    @param vector where we want to find the max length
    @return the max length of an element of the vector
    '''
    maxim = 0
    i = 0
    for el in vector:
        if(len(el)>maxim):
            maxim = len(el)
    return maxim


def convert_to_categorical(vec):
    '''
    Method used to create the categorical shape of the label vector
    :param vec the label vector
    :return a np.array with the categorical shape of the input
    '''
    matrix = []
    for v in vec:
        a = to_categorical(v,num_classes=4)
        matrix.append(np.array(a))
    return np.array(matrix)


