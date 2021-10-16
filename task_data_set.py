# -*- coding:utf-8 -*-
# @project: PowerBert
# @filename: task_data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/5 10:02
"""
    文件说明:
            
"""
import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence
import random
from multiprocessing import Pool
import re
import jieba

logger = logging.getLogger(__name__)


class DefectDiagnosisDataSet(Dataset):
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, path_file=None, train_rate = 1.0, is_overwrite=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = {"一般": 0, "严重": 1, "危急": 2}
        self.data_set_name = data_set_name
        self.train_rate = train_rate
        
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        self.data_set = []
        label_number_dict = {"一般": 0, "严重": 0, "危急": 0}
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                sample = json.loads(line)
                if sample["label"] not in self.label_dict:
                    continue
                if "train" in self.data_set_name:
                    if label_number_dict[sample["label"]] > int(self.train_rate * 200):
                        continue
                    else:
                        label_number_dict[sample["label"]] += 1
                input_ids, attention_mask, label = self.convert_feature(sample)
                self.data_set.append({"input_ids": input_ids, "attention_mask": attention_mask, "label": label})
        return self.data_set

    def convert_feature(self, sample):
        label = self.label_dict[sample["label"]]
        tokens = self.tokenizer.tokenize(sample["text"])
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func_defect_diagnosis(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["attention_mask"]
        labels_temp = instance["label"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        labels_list.append(labels_temp)
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "label": torch.tensor(labels_list, dtype=torch.long)}


class EntityExtractDataSet(Dataset):
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, path_file=None, is_overwrite=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                sample = json.loads(line)
                input_ids, attention_mask, labels, tokens = self.convert_feature(sample)
                self.data_set.append(
                    {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "tokens": tokens})
        return self.data_set

    def convert_feature(self, sample):
        tokens = []
        for t in sample["text"]:
            token = self.tokenizer.tokenize(t)
            if len(token) == 0:
                tokens.append("[unused1]")
            else:
                tokens.extend(token)
        labels = [self.label_dict[l] for l in sample["label"]]
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
            labels = labels[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        labels = [0] + labels + [0]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, labels, tokens

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func_entity_extract(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, labels_list = [], [], []
    tokens_list = []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["attention_mask"]
        labels_temp = instance["labels"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        labels_list.append(torch.tensor(labels_temp, dtype=torch.long))
        tokens_list.append(instance["tokens"])
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=0),
            "tokens": tokens_list}


class DefectExtractDataSet(Dataset):
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, label1_dict_path, label2_dict_path,
                 label3_dict_path, label4_dict_path, path_file=None, is_overwrite=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(label1_dict_path, "r", encoding="utf-8") as fh:
            l_list = [l.strip() for l in fh.readlines()]
            self.label1_dict = dict(zip(l_list, list(range(0, len(l_list)))))
        logger.info("label1_dict is {}".format(self.label1_dict))
        with open(label2_dict_path, "r", encoding="utf-8") as fh:
            l_list = [l.strip() for l in fh.readlines()]
            self.label2_dict = dict(zip(l_list, list(range(0, len(l_list)))))
        logger.info("label2_dict is {}".format(self.label2_dict))
        with open(label3_dict_path, "r", encoding="utf-8") as fh:
            l_list = [l.strip() for l in fh.readlines()]
            self.label3_dict = dict(zip(l_list, list(range(0, len(l_list)))))
        logger.info("label3_dict is {}".format(self.label3_dict))
        with open(label4_dict_path, "r", encoding="utf-8") as fh:
            l_list = [l.strip() for l in fh.readlines()]
            self.label4_dict = dict(zip(l_list, list(range(0, len(l_list)))))
        logger.info("label4_dict is {}".format(self.label4_dict))
        self.label5_dict = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
        logger.info("label5_dict is {}".format(self.label5_dict))
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                sample = json.loads(line)
                input_ids, attention_mask, label1, label2, label3, label4, label5_start, label5_end, tokens = self.convert_feature(
                    sample)
                self.data_set.append(
                    {"input_ids": input_ids, "attention_mask": attention_mask, "label1": label1, "label2": label2,
                     "label3": label3, "label4": label4, "label5_start": label5_start, "label5_end": label5_end, "tokens": tokens})
        return self.data_set

    def convert_feature(self, sample):
        label5_start = sample["text"].find(sample["label5"])
        label5_end = label5_start + len(sample["label5"])
        tokens = []
        for t in sample["text"]:
            token = self.tokenizer.tokenize(t)
            if len(token) == 0:
                tokens.append("[unused1]")
            else:
                tokens.extend(token)
        label1 = self.label1_dict[sample["label1"]]
        label2 = self.label2_dict[sample["label2"]]
        label3 = self.label3_dict[sample["label3"]]
        label4 = self.label4_dict[sample["label4"]]
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        label5_start = label5_start + 1
        label5_end = label5_end + 1
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, label1, label2, label3, label4, label5_start, label5_end, tokens

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func_defect_extract(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, label5_start_list, label5_end_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
    tokens_list = []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["attention_mask"]
        label1_temp = instance["label1"]
        label2_temp = instance["label2"]
        label3_temp = instance["label3"]
        label4_temp = instance["label4"]
        label5_start_temp = instance["label5_start"]
        label5_end_temp = instance["label5_end"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        label1_list.append(label1_temp)
        label2_list.append(label2_temp)
        label3_list.append(label3_temp)
        label4_list.append(label4_temp)
        label5_start_list.append(label5_start_temp)
        label5_end_list.append(label5_end_temp)
        tokens_list.append(instance["tokens"])
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "label1": torch.tensor(label1_list, dtype=torch.long),
            "label2": torch.tensor(label2_list, dtype=torch.long),
            "label3": torch.tensor(label3_list, dtype=torch.long),
            "label4": torch.tensor(label4_list, dtype=torch.long),
            "label5_start": torch.tensor(label5_start_list, dtype=torch.long),
            "label5_end": torch.tensor(label5_end_list, dtype=torch.long),
            "tokens": tokens_list}
