#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2017-03-29 14:06
# @Author  : Zhang Du
# @File    : BTransR_train.py
import numpy as np
from constants import *
import json
import random

class BTransR:
    def __init__(self, step = 0.001):
        def load_aligned_triples():
            with open(path_aligned_triples, mode='r', encoding='utf8') as file:
                # aligned triples: [ZH_h, ZH_t, ZH_r, ZH_h_index, ZH_t_index, ZH_r_index,
                #                   EN_h, EN_t, EN_r, EN_h_index, EN_t_index, EN_r_index]
                self.aligned_triples = json.load(file)

        def find_unique_zh_en_triples():
            def delete_duplicate_triples(list_triples):
                if len(list_triples) <= 1:
                    return list_triples
                list_triples.sort(key=lambda x: (x[3], x[4], x[5]))
                index = 0
                for i in range(1, len(list_triples)):
                    if list_triples[index] != list_triples[i]:
                        index = index + 1
                        list_triples[index] = list_triples[i]
                return list_triples[0: index + 1]

            # extract Chinese and English triples in aligned triples and delete duplication
            for i in self.aligned_triples:
                temp_zh = i[0: 6]   # ZH part
                temp_en = i[6: 12]  # EN part
                self.Chinese_triples.append(temp_zh)
                self.English_triples.append(temp_en)
            self.Chinese_triples = delete_duplicate_triples(self.Chinese_triples)
            self.English_triples = delete_duplicate_triples(self.English_triples)

        def subscript_zh_en_triples():
            # {num: [h, t, r, h_num, t_num, r_num], …, }
            for i in range(len(self.Chinese_triples)):
                self.zh_subscript_triples[i] = self.Chinese_triples[i]
            for i in range(len(self.English_triples)):
                self.en_subscript_triples[i] = self.English_triples[i]

        def zh_en_dict_aligned_triples():
            # self.zh_subscript_triples = {Chinese triple: [English triples,…,], … , }
            # self.en_subscript_triples = {English triple: [Chinese triples, …,], …, }
            for i in self.aligned_triples:
                temp_zh = tuple(i[0: 6])    # cannot hash a list
                temp_en = i[6: 12]
                if self.zh_dict_aligned_triples.get(temp_zh) == None:
                    self.zh_dict_aligned_triples[temp_zh] = [temp_en]
                else:
                    self.zh_dict_aligned_triples[temp_zh].append(temp_en)

            for i in self.aligned_triples:
                temp_zh = i[0: 6]
                temp_en = tuple(i[6: 12])   # cannot hash a list
                if self.en_dict_aligned_triples.get(temp_en) == None:
                    self.en_dict_aligned_triples[temp_en] = [temp_zh]
                else:
                    self.en_dict_aligned_triples[temp_en].append(temp_zh)

        def num2embedding():
            for i in range(len(path_vec)):
                with open(path_vec[i], mode='r', encoding='utf8') as file:
                    while True:
                        line = file.readline()
                        if line:
                            vectors = line.split('\t')
                            vectors = vectors[0: -1]
                            for j in range(len(vectors)):
                                vectors[j] = float(vectors[j])
                            # convert to numpy and transpose to column vector
                            vectors = np.array(vectors).transpose()
                            self.num_vector_list[i].append(vectors)
                        else:
                            break

        # BTransR basic attibutes
        self.dimension = 50
        self.step = step
        self.margin = 1
        self.aligned_triples = []
        load_aligned_triples()
        self.BTransR_length = len(self.aligned_triples)
        self.nbatches = 1
        self.BTransR_batch = int(self.BTransR_length / self.nbatches)
        self.BTransR_train_times = 3000
        self.loss = 0
        # assistant attributes
            # find corrupted triples
        self.Chinese_triples = []
        self.English_triples = []
        self.zh_subscript_triples = {}
        self.en_subscript_triples = {}
        find_unique_zh_en_triples()
        self.length_zh_triples = len(self.Chinese_triples)
        self.length_en_triples = len(self.English_triples)
        subscript_zh_en_triples()
        self.zh_dict_aligned_triples = {}
        self.en_dict_aligned_triples = {}
        zh_en_dict_aligned_triples()
            # transition between num and embeddings
        self.num_vector_entities_en = []
        self.num_vector_relations_en = []
        self.num_vector_entities_zh = []
        self.num_vector_relations_zh = []
        self.num_vector_list = [self.num_vector_entities_en, self.num_vector_relations_en,
                                self.num_vector_entities_zh, self.num_vector_relations_zh]
        num2embedding()
        self.matrix = []
        for i in range(len(self.num_vector_relations_en)):
            self.matrix.append(np.eye(self.dimension)) 
        self.loss_list = []
        self.min_loss = 9999


    def norm(self, t3_h, t3_t, t3_r, corrupt_is_zh):
        ''' 
        ||M*h||2 <= 1   ||M*t||2 <= 1 
        '''
        def norm_specific(entity_embedding):
            lambda_step = 1
            while True:
                x = 0
                mul = self.matrix[t3_r].dot(entity_embedding)
                for i in mul:
                    # x += i ** 0.5
                    x += i * i
                x = x ** 0.5

                if x > 1:
                    for ii in range(self.dimension):
                        tmp = self.matrix[t3_r][ii, 0: self.dimension].dot(entity_embedding)
                        for jj in range(self.dimension):
                            self.matrix[t3_r][ii][jj] -= self.step * lambda_step * tmp * entity_embedding[jj]
                            # entity_embedding[jj] -= self.step * lambda_step * tmp * self.matrix_head[ii][[jj]]
                else:
                    break

        if corrupt_is_zh:
            pass
        else:
            norm_specific(self.num_vector_entities_en[t3_h])
            norm_specific(self.num_vector_entities_en[t3_t])



    def BTransR(self):
        for epoch in range(self.BTransR_train_times):
            self.loss = 0
            for batch in range(self.nbatches):
                for k in range(self.BTransR_batch):
                    i = random.randint(0, self.BTransR_length - 1)  # randomize an aligned triples
                    random_num = random.randint(1, 1000)
                    if random_num <= 500:
                        corrupt_is_zh = True
                        corrupt_tri = self.corrupt_triples_former(self.aligned_triples[i])
                        # corrupted aligned triples, substitute a random triples
                        self.train_trans(self.aligned_triples[i][3], self.aligned_triples[i][4], self.aligned_triples[i][5], 
                                         self.aligned_triples[i][9], self.aligned_triples[i][10], self.aligned_triples[i][11], 
                                         corrupt_tri[0], corrupt_tri[1], corrupt_tri[2], 
                                         self.aligned_triples[i][9], self.aligned_triples[i][10], self.aligned_triples[i][11])
                    else:
                        corrupt_is_zh = False
                        corrupt_tri = self.corrupt_triples_latter(self.aligned_triples[i])
                        self.train_trans(self.aligned_triples[i][3], self.aligned_triples[i][4], self.aligned_triples[i][5], 
                                         self.aligned_triples[i][9], self.aligned_triples[i][10], self.aligned_triples[i][11], 
                                         self.aligned_triples[i][3], self.aligned_triples[i][4], self.aligned_triples[i][5],
                                         corrupt_tri[0], corrupt_tri[1], corrupt_tri[2])

                    self.norm(corrupt_tri[0], corrupt_tri[1], corrupt_tri[2], corrupt_is_zh)
            print("Step:", self.step, "epoch:", epoch, "loss:", self.loss)
            self.loss_list.append(self.loss)
            if min(self.min_loss, self.loss) == self.loss:
                self.min_loss = self.loss
                self.out_BTransR()
                print("Current min_loss is", self.min_loss, "In epoch", epoch)

    def out_BTransR(self):
        for i in range(len(self.num_vector_relations_en)):
            self.matrix[i].tofile(path_BTransR_Matrix[i])

    def train_trans(self, t1_h, t1_t, t1_r, t2_h, t2_t, t2_r, t3_h, t3_t, t3_r, t4_h, t4_t, t4_r):
        sum1 = self.calc_sum(t1_h, t1_t, t1_r, t2_h, t2_t, t2_r)
        sum2 = self.calc_sum(t3_h, t3_t, t3_r, t4_h, t4_t, t4_r)
        if sum1 + self.margin > sum2:
            self.loss += self.margin + sum1 - sum2
            self.gradient(t1_h, t1_t, t1_r, t2_h, t2_t, t2_r, -1, 1)
            self.gradient(t3_h, t3_t, t3_r, t4_h, t4_t, t4_r, 1, 1)

    def calc_sum(self, t1_h, t1_t, t1_r, t2_h, t2_t, t2_r):
        t1_h_embedding = self.num_vector_entities_zh[t1_h]
        t1_t_embedding = self.num_vector_entities_zh[t1_t]
        t1_r_embedding = self.num_vector_relations_zh[t1_r]
        t2_h_embedding = self.num_vector_entities_en[t2_h]
        t2_t_embedding = self.num_vector_entities_en[t2_t]
        t2_r_embedding = self.num_vector_relations_en[t2_r]

        head = self.matrix[t2_r].dot(t2_h_embedding) - t1_h_embedding
        head_fabs = np.fabs(head)
        head_sum = np.sum(head_fabs)

        tail = self.matrix[t2_r].dot(t2_t_embedding) - t1_t_embedding
        tail_fabs = np.fabs(tail)
        tail_sum = np.sum(tail_fabs)

        relation = t2_r_embedding - t1_r_embedding
        relation_fabs = np.fabs(relation)
        relation_sum = np.sum(relation_fabs)

        return head_sum + tail_sum + relation_sum

    def gradient(self, t1_h, t1_t, t1_r, t2_h, t2_t, t2_r, belta, same):
        t1_h_embedding = self.num_vector_entities_zh[t1_h]
        t1_t_embedding = self.num_vector_entities_zh[t1_t]
        t1_r_embedding = self.num_vector_relations_zh[t1_r]
        t2_h_embedding = self.num_vector_entities_en[t2_h]
        t2_t_embedding = self.num_vector_entities_en[t2_t]
        t2_r_embedding = self.num_vector_relations_en[t2_r]

        for ii in range(self.dimension):
            x_head = t2_h_embedding[ii] - self.matrix[t2_r][ii, 0: self.dimension].dot(t1_h_embedding)
            if x_head > 0:
                x_head = belta * self.step
            else:
                x_head = -belta * self.step
            for jj in range(self.dimension):
                self.matrix[t2_r][ii][jj] -= x_head * (t1_h_embedding[jj] - t2_h_embedding[jj])

            x_tail = t2_t_embedding[ii] - self.matrix[t2_r][ii, 0: self.dimension].dot(t1_t_embedding)
            if x_tail > 0:
                x_tail = belta * self.step
            else:
                x_tail = -belta * self.step
            for jj in range(self.dimension):
                self.matrix[t2_r][ii][jj] -= x_tail * (t1_t_embedding[jj]- t2_t_embedding[jj])

            x_relation = t2_r_embedding[ii] - t1_r_embedding[ii]
            if x_relation > 0:
                x_relation = belta * self.step
            else:
                x_relation = -belta * self.step
            for jj in range(self.dimension):
                self.matrix[t2_r][ii][jj] -= x_relation * (t1_r_embedding[jj]- t2_r_embedding[jj])

    # randomize a Chinese triple, guarantee this pair of triples is not aligned triples
    # otherwise randomize again.
    def corrupt_triples_former(self, aligned_triples):
        aligned_en_triples = aligned_triples[6: 12]
        while True:
            rand_zh_subscript = random.randint(0, self.length_zh_triples - 1)    # randomize the subscript of a Chinese triple
            rand_Chinese_triple = self.zh_subscript_triples[rand_zh_subscript]
            aligned_zh_triples = self.en_dict_aligned_triples[tuple(aligned_en_triples)]
            if rand_Chinese_triple not in aligned_zh_triples:
                return rand_Chinese_triple[3: 6]

    # randomize an English triple, guarantee this pair of triples is not aligned triples
    # otherwise randomize again.
    def corrupt_triples_latter(self, aligned_triples):
        aligned_zh_triples = aligned_triples[0: 6]
        while True:
           # randomize the subscript of a English triple
            rand_en_subscript = random.randint(0, self.length_en_triples - 1)  
            # use subscript to find the original English triple
            rand_English_triple = self.en_subscript_triples[rand_en_subscript]   
            # find English triples according to the original Chinese triple
            aligned_en_triples = self.zh_dict_aligned_triples[tuple(aligned_zh_triples)]
            # ensure the newly randomized triple is not in the original En and Zh triples set
            if rand_English_triple not in aligned_en_triples:
                return rand_English_triple[3: 6] # return the subscription of corrupt English triple


if __name__ == '__main__':
    timenow("Program begin.")
    test = BTransR()
    test.BTransR()
    timenow("Program end.")
