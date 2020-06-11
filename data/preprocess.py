# -*- coding:utf-8 -*-
import pandas as pd
import os
import torch
import numpy as np
import scipy.sparse as sp
import time
from functools import partial
import re
import string
import gc
import pickle as pkl
from tqdm import tqdm

class ItemLens(object):
    def __init__(self, neg_size=99):
        users = []
        items = []
        ratings = []
        with open('epinion_rating_with_timestamp.txt') as f:
            for l in f:
                user_id, item_id, _, rating, _, timestamp = [int(float(_)) for _ in l.strip().split('  ')]
                ratings.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)
        item_count = ratings['item_id'].value_counts()
        item_count.name = 'item_count'
        ratings = ratings.join(item_count, on='item_id')
        self.ratings = ratings

        # determine test and validation set
        self.ratings['timerank'] = self.ratings.groupby('user_id')['timestamp'].rank(ascending=False).astype('int')
        self.ratings['test_mask'] = (self.ratings['timerank'] == 1)
        self.ratings['valid_mask'] = (self.ratings['timerank'] == 2)

        # remove items that only appear in validation and test set
        items_selected = self.ratings[self.ratings['timerank'] > 2]['item_id'].unique()
        self.ratings = self.ratings[self.ratings['item_id'].isin(items_selected)].copy()
        users_selected = self.ratings[self.ratings['timerank'] > 2]['user_id'].unique()
        self.ratings = self.ratings[self.ratings['user_id'].isin(users_selected)].copy()

        # drop users and movies which do not exist in ratings
        self.users = self.ratings[['user_id']].drop_duplicates(subset=['user_id'],keep='first')
        # self.users = self.users[self.users['id'].isin(self.ratings['user_id'])]
        self.items = self.ratings[['item_id']].drop_duplicates(subset=['item_id'],keep='first')
        # self.movies = self.movies[self.movies['id'].isin(self.ratings['movie_id'])]
        self.rates = self.ratings[['rating']].drop_duplicates(subset=['rating'],keep='first')

        self.user_ids_invmap = {u: i for i, u in enumerate(self.users['user_id'])}
        self.item_ids_invmap = {m: i for i, m in enumerate(self.items['item_id'])}
        self.rating_ids_invmap = {r: i for i, r in enumerate(self.rates['rating'])}

        self.ratings['user_idx'] = self.ratings['user_id'].apply(lambda x: self.user_ids_invmap[x])
        self.ratings['item_idx'] = self.ratings['item_id'].apply(lambda x: self.item_ids_invmap[x])
        self.ratings['rating_idx'] = self.ratings['rating'].apply(lambda x: self.rating_ids_invmap[x])

        # parse item features
        item_data = {}
        self.num_items = len(self.items)
        self.num_users = len(self.users)
        print(self.num_items)
        print(self.num_users)
        # unobserved items for each user in training set
        self.neg_train = [None] * len(self.users)
        # negative examples for validation and test for evaluating ranking
        self.neg_valid = np.zeros((len(self.users), neg_size), dtype='int64')
        self.neg_test = np.zeros((len(self.users), neg_size), dtype='int64')
        rating_groups = self.ratings.groupby('user_idx')
        # print(len(self.ratings['item_idx']))
        # print(len(self.ratings['user_idx']))
        # print(self.ratings[1000:1200])
        # print(self.ratings['item_idx'][1188])

        for u in tqdm(range(len(self.users))):
            interacted_items = self.ratings['item_idx'].iloc[rating_groups.indices[u]]
            timerank = self.ratings['timerank'].iloc[rating_groups.indices[u]]

            interacted_items_valid = interacted_items[timerank >= 1]
            # print(len(interacted_items_valid))
            neg_samples = np.setdiff1d(np.arange(len(self.items)), interacted_items_valid)
            # self.neg_train[u] = neg_samples
            self.neg_valid[u] = np.random.choice(neg_samples, neg_size)

            del neg_samples

            interacted_items_test = interacted_items[timerank >= 1]
            neg_samples = np.setdiff1d(np.arange(len(self.items)), interacted_items_test)
            self.neg_test[u] = np.random.choice(neg_samples, neg_size)

            del neg_samples

            gc.collect()


if __name__ == "__main__":
    history_u, history_i, history_ur, history_ir = {}, {}, {}, {}
    data = ItemLens()
    ratings = data.ratings
    ratings_train = ratings[~(ratings['valid_mask'] | ratings['test_mask'])]
    user_idx = ratings_train['user_idx']
    item_idx = ratings_train['item_idx']
    rating_idx = ratings_train['rating_idx']
    ratings_set = set()

    for u, i, r in zip(user_idx, item_idx, rating_idx):
        ratings_set.add(r)
        if u not in history_u:
            history_u[u] = []
            history_ur[u] = []
        history_u[u].append(i)
        history_ur[u].append(r)

        if i not in history_i:
            history_i[i] = []
            history_ir[i] = []
        history_i[i].append(u)
        history_ir[i].append(r)

    assert data.num_users == len(history_u)
    assert data.num_items == len(history_i)

    train_u, train_i, train_r = [], [], []
    valid_u, valid_i, valid_r = [], [], []
    test_u, test_i, test_r = [], [], []
    # ratings_train = ratings[~(ratings['valid_mask'] | ratings['test_mask'])]
    ratings_valid = ratings[ratings['valid_mask']]
    ratings_test = ratings[ratings['test_mask']]
    for u, i, r in zip(ratings_train['user_idx'], ratings_train['item_idx'], ratings_train['rating']):
        train_u.append(u)
        train_i.append(i)
        train_r.append(r)

    for u, i, r in zip(ratings_valid['user_idx'], ratings_valid['item_idx'], ratings_valid['rating']):
        valid_u.append(u)
        valid_i.append(i)
        valid_r.append(r)

    for u, i, r in zip(ratings_test['user_idx'], ratings_test['item_idx'], ratings_test['rating']):
        test_u.append(u)
        test_i.append(i)
        test_r.append(r)

    social_neighbor = {}
    num_neighbor = 0
    with open('epinion_trust.txt', 'r', encoding='utf-8') as f:
        for line in f:
            user1id, user2id = line.strip().split('  ')
            user1id, user2id = int(float(user1id)), int(float(user2id))
            if user1id not in data.user_ids_invmap or user2id not in data.user_ids_invmap:
                continue
            user1id_idx, user2id_idx = data.user_ids_invmap[user1id], data.user_ids_invmap[user2id]
            if user1id_idx not in social_neighbor:
                social_neighbor[user1id_idx] = []
            social_neighbor[user1id_idx].append(user2id_idx)
            num_neighbor += 1

    if len(social_neighbor) != data.num_users:
        print('# of users have friends: '+str(len(social_neighbor)))
        nofriends_users = np.setdiff1d(np.arange(data.num_users), list(social_neighbor.keys()))
        for u in nofriends_users:
            social_neighbor[u] = [u]
            num_neighbor += 1

    valid_rank = {}
    test_rank = {}

    for u, i, r in zip(valid_u, valid_i, valid_r):
        if r < 3:
            continue
        neg_samples = data.neg_valid[u]
        valid_rank[u] = {'pos': i, 'neg':neg_samples}

    for u, i, r in zip(test_u, test_i, test_r):
        if r < 3:
            continue
        neg_samples = data.neg_test[u]
        test_rank[u] = {'pos': i, 'neg':neg_samples}
        

    f = open('epinion_dataset.pkl', 'wb')
    data_content = (history_u, history_i, history_ur, history_ir, train_u, train_i, train_r, valid_u, valid_i, valid_r,
                     test_u, test_i, test_r, social_neighbor, list(ratings_set))
    pkl.dump(data_content, f)
    f.close()

    f = open('epinion_dataset_rank.pkl', 'wb')
    pkl.dump((valid_rank, test_rank), f)
    f.close()

    print('# of users: '+str(data.num_users))
    print('# of items: '+str(data.num_items))
    print('# of train interactions: '+str(len(train_u)))
    print('# of valid interactions: '+str(len(valid_u)))
    print('# of test interactions: '+str(len(test_u)))
    print('# of rating: '+str(len(list(ratings_set))))
    print('# of valid rank interactions: '+str(len(valid_rank)))
    print('# of test rank interactions: '+str(len(test_rank)))
    print('# of neighbors: '+str(num_neighbor))
