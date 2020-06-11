# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import os
import io
import sys
import logging
from utils import config, metrics
from model import GraphRec

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, train_loader, optimizer, epoch, val_loader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        batch_nodes_u, batch_nodes_i, batch_ratings = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_i.to(device), batch_ratings.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            # print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
            #     epoch, i, running_loss / 10, best_rmse, best_mae))
            logger.info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 10))
            running_loss = 0.0
    
    hits = rank_test(model, device, val_loader)

    return hits

def test(model, device, test_loader):
    model.eval()
    pred = []
    target = []
    with torch.no_grad():
        for test_u, test_i, test_ratings in test_loader:
            test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
            scores = model(test_u, test_i)
            pred.append(list(scores.cpu().numpy()))
            target.append(list(test_ratings.cpu().numpy()))
    pred = np.array(sum(pred, []))
    target = np.array(sum(target, []))
    rmse = sqrt(mean_squared_error(pred, target))
    mae = mean_absolute_error(pred, target)

    return rmse, mae

def rank_test(model, device, test_data):
    model.eval()
    rank_list = []
    for u in test_data:
        item = test_data[u]
        neg_list = item['neg'].tolist()
        user = torch.LongTensor([u]*len(neg_list)).to(device)
        pos = torch.LongTensor([item['pos']]).to(device)
        neg = torch.LongTensor(neg_list).to(device)
        with torch.no_grad():
            scores_neg = model(user, neg)
            scores_pos = model(user[:1], pos)
            rank = np.argsort(-np.hstack((scores_pos.cpu().numpy(), scores_neg.cpu().numpy())))
            rank = rank[:10]
            rank_list.append(int(0 in rank))

    # logger.info("Hits@10:%.4f " % (np.mean(rank_list)))

    return np.mean(rank_list)


def main():
    args = config.config()
    logger.info(args.embed_dim)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    embed_dim = args.embed_dim

    dir_data = 'data/'+args.dataset+'_dataset'
    path_data = dir_data + ".pkl"
    data_file = open(path_data, 'rb')
    history_u, history_i, history_ur, history_ir, train_u, train_i, train_r, valid_u, valid_i, valid_r,\
                 test_u, test_i, test_r, social_neighbor, ratings = pickle.load(data_file)

    path_data_rank = dir_data + "_rank.pkl"
    rank_data = open(path_data_rank, 'rb')
    valid_rank_data, test_rank_data = pickle.load(rank_data)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_i),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_i),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_i),
                                             torch.FloatTensor(test_r))
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u.__len__()
    num_items = history_i.__len__()
    num_ratings = ratings.__len__()

    # model
    graphrec = GraphRec(num_users, num_items, num_ratings, history_u, history_i, history_ur,\
                                     history_ir, embed_dim, social_neighbor, cuda=device).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    # best_rmse = 9999.0
    # best_mae = 9999.0
    best_hits = 0.0
    endure_count = 0
    best_test_hits = 0

    for epoch in range(1, args.epochs+1):
        val_hits = train(graphrec, device, train_loader, optimizer, epoch, valid_rank_data)
        
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping
        if best_hits < val_hits:
            best_hits = val_hits
            endure_count = 0
        else:
            endure_count += 1
        logger.info("val HITS@10:%.4f " % (val_hits))

        test_hits = rank_test(graphrec, device, test_rank_data)
        if test_hits > best_test_hits:
            best_test_hits = test_hits
        logger.info("best test HITS@10:%.4f " % (best_test_hits))

        if endure_count > 5:
            logger.info("early stopping...")
            break

    test_hits = rank_test(graphrec, device, test_rank_data)
    logger.info("test HITS@10:%.4f " % (test_hits))
    logger.info("best test HITS@10:%.4f " % (best_test_hits))

    # test_rmse, test_mae = test(graphrec, device, test_loader)
    # logger.info("test rmse: %.4f, test mae:%.4f " % (test_rmse, test_mae))


if __name__ == "__main__":
    main()
