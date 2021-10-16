# -*- coding:utf-8 -*-
# @project: PowerBert
# @filename: run_entity_extract
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/5 13:41
"""
    文件说明:
            
"""

import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import BertTokenizer
from task_data_set import EntityExtractDataSet, collate_func_entity_extract
from model import EntityExtractModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

import json
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, tokenizer, args):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        tokenizer: 分词器
        args: 训练参数配置信息

    Returns:

    """
    tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_data = EntityExtractDataSet(tokenizer, args.max_len, args.data_dir, "train_entity_extract",
                                      args.train_file_path)
    dev_data = EntityExtractDataSet(tokenizer, args.max_len, args.data_dir, "dev_entity_extract",
                                    args.dev_file_path)
    test_data = EntityExtractDataSet(tokenizer, args.max_len, args.data_dir, "test_entity_extract",
                                     args.test_file_path)

    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func_entity_extract)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    torch.cuda.empty_cache()
    model.train()
    tr_loss, logging_loss, max_acc = 0.0, 0.0, 0.0
    global_step = 0
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model.forward(input_ids, attention_mask, labels)
            loss = outputs[0]
            tr_loss += loss.item()
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                        (args.logging_steps * args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss

                if args.save_model_steps > 0 and global_step % args.save_model_steps == 0:
                    eval_loss, eval_recall, eval_precision, eval_f1, _ = evaluate(model, device, dev_data, args)
                    logger.info(
                        "eval_loss is {}, eval_recall is {} , eval_precision is {} ,eval_f1 is {}".format(eval_loss,
                                                                                                          eval_recall,
                                                                                                          eval_precision,
                                                                                                          eval_f1))
                    tb_write.add_scalar("eval_loss", eval_loss, global_step)
                    tb_write.add_scalar("eval_recall", eval_recall, global_step)
                    tb_write.add_scalar("eval_precision", eval_precision, global_step)
                    tb_write.add_scalar("eval_f1", eval_f1, global_step)

                    test_loss, test_recall, test_precision, test_f1, _ = evaluate(model, device, test_data, args)
                    logger.info(
                        "test_loss is {}, test_recall is {} , test_precision is {} , test_f1 is {}".format(test_loss,
                                                                                                           test_recall,
                                                                                                           test_precision,
                                                                                                           test_f1))
                    tb_write.add_scalar("test_loss", test_loss, global_step)
                    tb_write.add_scalar("test_recall", test_recall, global_step)
                    tb_write.add_scalar("test_precision", test_precision, global_step)
                    tb_write.add_scalar("test_f1", test_f1, global_step)
                    if eval_f1 >= max_acc:
                        max_acc = eval_f1
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        json_output_dir = os.path.join(output_dir, "json_data.json")
                        fin = open(json_output_dir, "w", encoding="utf-8")
                        fin.write(json.dumps(
                            {"eval_recall": float(eval_recall), "eval_precision": float(eval_precision),
                             "eval_f1": float(eval_f1), "test_recall": float(test_recall),
                             "test_precision": float(test_precision), "test_f1": float(test_f1)},
                            ensure_ascii=False, indent=4))
                        fin.close()
                    model.train()


def evaluate(model, device, dev_data, args):
    test_sampler = SequentialSampler(dev_data)
    test_data_loader = DataLoader(dev_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_entity_extract)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    eval_loss = 0.0
    true_label = []
    pre_label = []
    tokens = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["labels"].to(device)
            tokens.extend(batch["tokens"])
            [loss, tags, _] = model.forward(input_ids, attention_mask, label)
            eval_loss += loss.item()
            true_label.extend(label.cpu().numpy().tolist())
            pre_label.extend(tags[0].cpu().numpy().tolist())
    r_list, p_list, f1_list = [], [], []
    s_list = []
    for tag, label, token in zip(pre_label, true_label, tokens):
        pred_terms, real_terms = get_entity(tag, label, token)
        r, p, f1 = get_recall_precision_f1(pred_terms, real_terms)
        r_list.append(r)
        p_list.append(p)
        f1_list.append(f1)
        s_list.append({"token": token, "pred_terms": pred_terms, "real_terms": real_terms})
    r = np.mean(np.array(r_list))
    p = np.mean(np.array(p_list))
    f1 = np.mean(np.array(f1_list))

    eval_loss = eval_loss / len(test_data_loader)
    return eval_loss, r, p, f1, s_list


def test(model, device, tokenizer, args):
    test_data = EntityExtractDataSet(tokenizer, args.max_len, args.data_dir, "test_entity_extract",
                                     args.test_file_path)
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_entity_extract)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    eval_loss = 0.0
    true_label = []
    pre_label = []
    tokens = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["labels"].to(device)
            tokens.extend(batch["tokens"])
            [loss, tags, _] = model.forward(input_ids, attention_mask, label)
            eval_loss += loss.item()
            true_label.extend(label.cpu().numpy().tolist())
            pre_label.extend(tags[0].cpu().numpy().tolist())
    r_list, p_list, f1_list = [], [], []
    s_list = []
    for tag, label, token in zip(pre_label, true_label, tokens):
        pred_terms, real_terms = get_entity(tag, label, token)
        r, p, f1 = get_recall_precision_f1(pred_terms, real_terms)
        r_list.append(r)
        p_list.append(p)
        f1_list.append(f1)
        s_list.append({"token": token, "pred_terms": pred_terms, "real_terms": real_terms})
    r = np.mean(np.array(r_list))
    p = np.mean(np.array(p_list))
    f1 = np.mean(np.array(f1_list))

    eval_loss = eval_loss / len(test_data_loader)
    return eval_loss, r, p, f1, s_list


def get_recall_precision_f1(pred_terms, real_terms):
    s1, s2 = set(pred_terms), set(real_terms)
    r, p, f1 = 0.0, 0.0, 0.0
    t = len(s1.intersection(s2))
    if len(s1) == 0 and len(s2) == 0:
        return 1.0, 1.0, 1.0
    if len(s2) > 0:
        r = t / len(s2)
    if len(s1) > 0:
        p = t / len(s1)
    if (r + p) > 0.0:
        f1 = (2 * r * p) / (r + p)
    return r, p, f1


def get_entity(tag, label, token):
    label_dict = {0: "O", 1: "B", 2: "I", 3: "E", 4: "S"}
    pred_terms, real_terms = [], []

    tag = [label_dict[t] for t in tag]
    idx = 0
    term = ""
    s_idx = -1
    e_idx = -1
    while idx < len(token):
        if tag[idx] == "S":
            pred_terms.append((token[idx], idx, idx))
        elif tag[idx] == "B":
            term = ""
            term += token[idx]
            s_idx = idx
        elif tag[idx] == "I":
            if s_idx == -1:
                term = ""
            else:
                term += token[idx]
        elif tag[idx] == "E":
            if s_idx == -1:
                term = ""
                s_idx = -1
                e_idx = -1
            else:
                term += token[idx]
                e_idx = idx
                pred_terms.append((term, s_idx, e_idx))
        else:
            term = ""
            s_idx = -1
            e_idx = -1
        idx += 1

    label = [label_dict[t] for t in label]
    idx = 0
    term = ""
    s_idx = -1
    e_idx = -1
    while idx < len(token):
        if label[idx] == "S":
            real_terms.append((token[idx], idx, idx))
        elif label[idx] == "B":
            term = ""
            term += token[idx]
            s_idx = idx
        elif label[idx] == "I":
            if s_idx == -1:
                term = ""
            else:
                term += token[idx]
        elif label[idx] == "E":
            if s_idx == -1:
                term = ""
                s_idx = -1
                e_idx = -1
            else:
                term += token[idx]
                e_idx = idx
                real_terms.append((term, s_idx, e_idx))
        else:
            term = ""
            s_idx = -1
            e_idx = -1
        idx += 1
    return pred_terms, real_terms


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--train_file_path', default='task_data/train_entity_extract.json', type=str, help='')
    parser.add_argument('--dev_file_path', default='task_data/dev_entity_extract.json', type=str, help='')
    parser.add_argument('--test_file_path', default='task_data/test_entity_extract.json', type=str, help='')
    parser.add_argument('--vocab_path', default="chinese_L-12_H-768_A-12/vocab.txt", type=str, help='')
#     parser.add_argument('--pretrained_model_path', default="chinese_L-12_H-768_A-12/", type=str, help='')
#     parser.add_argument('--pretrained_model_path', default="output_dir_ori_bert_only_mlm/checkpoint-295000/", type=str, help='')
    parser.add_argument('--pretrained_model_path', default="../output_dir_ori_bert_only_mlm_v2/checkpoint-295000/", type=str, help='')
    parser.add_argument('--data_dir', default='task_data/', type=str, help='')
    parser.add_argument('--num_train_epochs', default=6, type=int, help='')
    parser.add_argument('--train_batch_size', default=8, type=int, help='')
    parser.add_argument('--test_batch_size', default=4, type=int, help='')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='')
    parser.add_argument('--save_model_steps', default=53, type=int, help='')
    parser.add_argument('--logging_steps', default=5, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='bp_output_dir/entity_extract_output_dir', type=str,
                        help='')
    parser.add_argument('--test_model_path', default='bp_output_dir/entity_extract_output_dir', type=str,
                        help='')
    parser.add_argument('--is_training', type=bool, default=True, help='')
    parser.add_argument('--is_testing', type=bool, default=False, help='')
    parser.add_argument('--seed', type=int, default=2020, help='')
    parser.add_argument('--max_len', type=int, default=256, help='')
    parser.add_argument('--num_labels', type=int, default=5, help='')
    return parser.parse_args()


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    model = EntityExtractModel.from_pretrained(args.pretrained_model_path, num_labels=args.num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.is_training:
        train(model, device, tokenizer, args)
    if args.is_testing:
        model = EntityExtractModel.from_pretrained(args.test_model_path, num_labels=args.num_labels)
        loss, r, p, f1, s_list = test(model, device, tokenizer, args)
        print("loss is {}, r is {}, p is {}, f1 is {}".format(loss, r, p, f1))


if __name__ == '__main__':
    main()
