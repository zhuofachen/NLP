from utils.cal_similarity import *
from data_preprocess.preprocess import *
import torch
from torchtext import data
import random
import numpy as np


#--------for reproducibility block-----------
# SEED = 2021
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)
# torch.use_deterministic_algorithms(True)
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     numpy.random.seed(worker_seed)
#     random.seed(worker_seed)
# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker
# )
#---------------------------------------------


# for loading data
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# #set batch size
# BATCH_SIZE = 64
#
# #Load an iterator
# train_iterator, valid_iterator = data.BucketIterator.splits(
#     (train_data, valid_data),
#     batch_size = BATCH_SIZE,
#     sort_key = lambda x: len(x.text),
#     sort_within_batch=True,
#     device = device)

if __name__ == "__main__":


    torch.backends.cudnn.deterministic = True
