#!/usr/bin/env python
# coding: utf-8

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score


def read_json(path):
    data = []
    with open(path, 'r',encoding='utf-8') as fp:
        for line in fp:
            data.append(json.loads(line))
    return data


class BaselineData(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_size = config.pad_size
        self.label2id = config.label2id
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = '[SEP]'.join([self.data[idx]['hashtag']] + self.data[idx]['comments'])

        input_ids, attention_mask = self.__convert_to_id__(sentence)

        if self.data[idx].get('attitudes'):
            label = self.__convert_label__(self.data[idx]['attitudes'])
            return torch.tensor(input_ids),                   torch.tensor(attention_mask),                   torch.tensor(label)
        else:
            return torch.tensor(input_ids),                   torch.tensor(attention_mask),
    def __convert_to_id__(self, sentence):
        ids = self.tokenizer.encode_plus(sentence)
        input_ids = self.__padding__(ids['input_ids'])
        attention_mask = self.__padding__(ids['attention_mask'])

        return input_ids, attention_mask
    
    def __convert_label__(self, label):
        onehot_label = [0] * 24
        for i in label:
            onehot_label[self.label2id[i]] = 1
        return onehot_label

    def __padding__(self, sentence):
        sentence = sentence[:self.pad_size]  # 长就截断
        sentence = sentence + [0] * (self.pad_size-len(sentence))  # 短就补充
        return sentence


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)
        self.fc = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.fc1 = nn.Linear(self.bert_config.hidden_size, config.label_num)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.bert(input_ids=x[0], attention_mask=x[1]).pooler_output
        x = self.sigmoid(self.fc1(self.act(self.fc(x))))
        return x


def train(config, dataset, model, optimizer, valid_dataset):
    max_scores = 0
    for epoch in range(config.epochs):
        with tqdm(total=len(dataset)) as pbar:
            for idx, data in enumerate(dataset):
                x = [data[0].long().to(config.device), data[1].long().to(config.device)]
                y = data[2].float().to(config.device)
                y_hat = model(x)
                loss = F.binary_cross_entropy(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss' : '{:.4f}'.format(loss)})
                pbar.update(1)

        scores = valid(config, valid_dataset, model)
        if scores >= max_scores:
            max_scores = scores
            saved_model = model
    return saved_model

def valid(config, dataset, model):
    true = []
    pred = []
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            x = [data[0].long().to(config.device), data[1].long().to(config.device)]
            y = data[2].float().to(config.device).view(-1, 24).tolist()
            y_hat = model(x).view(-1, 24).tolist()
            true.extend(y)
            pred.extend(y_hat)

    pred = [[1 if i>=0.5 else 0 for i in j] for j in pred]

    macro_f1 = f1_score(pred, true, average='macro')

    print('macro_f1: {:.4f}'.format(macro_f1))
    return macro_f1

def generate_test_result(config, dataset, model):
    with torch.no_grad():
        predict = []
        for idx, data in enumerate(dataset):
            x = [data[0].long().to(config.device), data[1].long().to(config.device)]
            predict.extend(model(x).view(-1, 24).tolist())
    with open('submit.txt', 'w', encoding='utf-8') as f:
        for i in range(len(predict)):
            line = []
            for j in range(len(predict[i])):
                if predict[i][j] >= 0.5:
                    line.append(config.id2label[j])
            f.write(' '.join([str(i)]+line))
            f.write('\n')


class Config():
    def __init__(self):
        self.pad_size = 510
        self.batch_size = 2
        self.epochs = 15
        self.PTM = 'bert-base-chinese'
        self.label_num = 24
        self.device = 'cuda:0'
        self.lr = 5e-5

        label_dic = ['[微笑]', '[嘻嘻]', '[笑cry]', '[怒]', '[泪]', '[允悲]', '[憧憬]', '[doge]', '[并不简单]', '[思考]', '[费解]', '[吃惊]', '[拜拜]', '[吃瓜]', '[赞]', '[心]', '[伤心]', '[蜡烛]', '[给力]', '[威武]', '[跪了]', '[中国赞]', '[给你小心心]', '[酸]']

        self.id2label = {k: v for k, v in enumerate(label_dic)}  # 用于标签的部分
        self.label2id = {v: k for k, v in enumerate(label_dic)}


config = Config()
train_data = read_json('train.json')
valid_data = read_json('valid.json')
test_data = read_json('test.json')

tokenizer = BertTokenizer.from_pretrained(config.PTM)

train_dataloader = DataLoader(BaselineData(train_data, tokenizer, config), batch_size=config.batch_size)
valid_dataloader = DataLoader(BaselineData(valid_data, tokenizer, config), batch_size=config.batch_size)
test_dataloader = DataLoader(BaselineData(test_data, tokenizer, config), batch_size=config.batch_size * 4)


model = Model(config).to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), config.lr)


# In[9]:


best_model = train(config, train_dataloader, model, optimizer, valid_dataloader)
generate_test_result(config, test_dataloader, best_model)

