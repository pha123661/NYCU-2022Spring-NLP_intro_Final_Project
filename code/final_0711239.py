#Author: 李勝維 符凱喻 許瀚宇 林俊宇
#Student ID: 0711239 0711278 0812501 0816038
#HW ID: final_project
#Due Date: 06/07/2022

import numpy as np
import random
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn.functional as F
import my_def
from bert_models import BertForTokenClassification
import math
from tqdm import tqdm, trange


# 設定亂數
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 參數雜項設定
num_train_epochs = 3
train_batch_size = 32
test_batch_size = 8
gradient_accumulation_steps = 1
max_seq_length = 128
warmup_proportion = 0.1
learning_rate = 5e-5

# 設定label 'O': not pun 'P': pun
# number要+1, 給padding用
label_list = ['O', 'P', '[CLS]', '[SEP]']
num_labels = len(label_list) + 1

# set bert model
bert_model = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)

# 把train data轉成example
train_examples = my_def.get_train_example()
# 計算train step數
num_train_optimization_steps = int(math.ceil(len(train_examples)
                                             / train_batch_size) / gradient_accumulation_steps) * num_train_epochs

# 取得pretrained model
cache_dir = 'for_cache/'
model = BertForTokenClassification.from_pretrained(bert_model,
                                                   cache_dir=cache_dir,
                                                   num_labels=num_labels,
                                                   max_seq_length=max_seq_length)

# 設定optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)


########################### train ######################

# example 轉 feature
train_features = my_def.convert_examples_to_features(
    train_examples, label_list, max_seq_length, tokenizer)

# feature 轉 tensordataset
all_input_ids = torch.tensor(
    [f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor(
    [f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor(
    [f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor(
    [f.label_ids for f in train_features], dtype=torch.long)
train_data = TensorDataset(
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

# sampler and loader
# sampler用random
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=train_batch_size)

# mode設成train
model.train()

# 開始epoch
for index in trange(int(num_train_epochs), desc='Epoch'):

    for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
        batch = tuple(t for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        loss, something_else = model(
            input_ids, segment_ids, input_mask, label_ids)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


############## evaluation ###############

# 取得test feature和submission
test_examples, submission = my_def.get_test_example_and_submission()
test_features = my_def.convert_examples_to_features(
    test_examples, label_list, max_seq_length, tokenizer)


# feature 轉 tensordataset
all_input_ids = torch.tensor(
    [f.input_ids for f in test_features], dtype=torch.long)
all_input_mask = torch.tensor(
    [f.input_mask for f in test_features], dtype=torch.long)
all_segment_ids = torch.tensor(
    [f.segment_ids for f in test_features], dtype=torch.long)
all_label_ids = torch.tensor(
    [f.label_ids for f in test_features], dtype=torch.long)
test_data = TensorDataset(all_input_ids, all_input_mask,
                          all_segment_ids, all_label_ids)


# sampler and loader
# 因為是test, 所以sampler必須用sequential
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(
    test_data, sampler=test_sampler, batch_size=test_batch_size)

# mode改成eval
model.eval()

# 和label_ids 一樣的標號
label_map = {i: label for i, label in enumerate(label_list, 1)}

# 存prediction : list of list of labels
all__pred = []

# evaluating
for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc='Evaluating'):
    input_ids = input_ids
    input_mask = input_mask
    segment_ids = segment_ids
    label_ids = label_ids

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask)

    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.numpy()
    input_mask = input_mask.numpy()
    for i, mask in enumerate(input_mask):
        pred_labels = []
        for j, m in enumerate(mask):
            if j == 0:
                continue
            if m:
                pred_labels.append(label_map[logits[i][j]])
            else:
                pred_labels.pop()
                all__pred.append(pred_labels)
                break

# 把pred結果放到submission
for idx, sentence_pred in enumerate(all__pred):
    try:
        submission.loc[idx, 'word_id'] = submission.loc[idx,
                                                        'text_id'] + '_' + str(sentence_pred.index('P') + 1)
    except:
        submission.loc[idx, 'word_id'] = submission.loc[idx,
                                                        'text_id'] + '_' + str(random.randint(1, len(sentence_pred)))

# export submission
submission.to_csv('submission.csv', index=False, encoding='utf-8')
