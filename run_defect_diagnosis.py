# -*- coding:utf-8 -*-
# @project: PowerBert
# @filename: run_defect_diagnosis
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/5 10:32
"""
    文件说明:
            
"""
import torch
import os
import random
import numpy as np
import time
import argparse
import logging
from transformers import BertTokenizer
from task_data_set import DefectDiagnosisDataSet, collate_func_defect_diagnosis
from model import DefectDiagnosisModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from task_data_helper import get_defect_diagnosis, split_defect_diagnosis_data

from sklearn.metrics import classification_report, f1_score, accuracy_score
import json
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, tokenizer, args, train_data_rate, ipc):
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
    train_data = DefectDiagnosisDataSet(tokenizer, args.max_len, args.data_dir, "train_defect_diagnosis",
                                        args.train_file_path, train_data_rate)
    dev_data = DefectDiagnosisDataSet(tokenizer, args.max_len, args.data_dir, "dev_defect_diagnosis",
                                      args.dev_file_path)
    test_data = DefectDiagnosisDataSet(tokenizer, args.max_len, args.data_dir, "test_defect_diagnosis",
                                       args.test_file_path)

    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func_defect_diagnosis)
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
            label = batch["label"].to(device)
            outputs = model.forward(input_ids, attention_mask, label)
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

#                 if args.save_model_steps > 0 and global_step % args.save_model_steps == 0:
#                     eval_loss, eval_acc, eval_f1 = evaluate(model, device, dev_data, args)
#                     logger.info("eval_loss is {}, eval_acc is {} , eval_f1 is {}".format(eval_loss, eval_acc, eval_f1))
#                     tb_write.add_scalar("eval_loss", eval_loss, global_step)
#                     tb_write.add_scalar("eval_acc", eval_acc, global_step)
#                     tb_write.add_scalar("eval_f1", eval_f1, global_step)
#                     test_loss, test_acc, test_f1 = evaluate(model, device, test_data, args)
#                     logger.info("test_loss is {}, test_acc is {} , test_f1 is {}".format(test_loss, test_acc, test_f1))
#                     tb_write.add_scalar("test_loss", test_loss, global_step)
#                     tb_write.add_scalar("test_acc", test_acc, global_step)
#                     tb_write.add_scalar("test_f1", test_f1, global_step)
    eval_loss, eval_acc, eval_f1 = evaluate(model, device, dev_data, args)
    logger.info("eval_loss is {}, eval_acc is {} , eval_f1 is {}".format(eval_loss, eval_acc, eval_f1))
    test_loss, test_acc, test_f1 = evaluate(model, device, test_data, args)
    logger.info("test_loss is {}, test_acc is {} , test_f1 is {}".format(test_loss, test_acc, test_f1))
    output_dir = os.path.join(args.output_dir, "checkpoint-{}-{}".format(ipc, train_data_rate))
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    json_output_dir = os.path.join(output_dir, "json_data.json")
    fin = open(json_output_dir, "w", encoding="utf-8")
    fin.write(json.dumps(
        {"eval_acc": float(eval_acc), "eval_f1": float(eval_f1), "test_acc": float(test_acc),
         "test_f1": float(test_f1), "ipc": int(ipc), "train_rate": float(train_data_rate)},
        ensure_ascii=False, indent=4))
    fin.close()
    model.train()


def evaluate(model, device, dev_data, args):
    test_sampler = SequentialSampler(dev_data)
    test_data_loader = DataLoader(dev_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_defect_diagnosis)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    eval_loss = 0.0
    true_label = []
    pre_label = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            [loss, predict_label, _] = model.forward(input_ids, attention_mask, label)
            eval_loss += loss.item()
            true_label.extend(label.cpu().numpy())
            pre_label.extend(predict_label.cpu().numpy())
    true_label = np.array(true_label)
    pre_label = np.array(pre_label)
    f1 = f1_score(true_label, pre_label, average='micro')
    acc = accuracy_score(true_label, pre_label)
    eval_loss = eval_loss / len(test_data_loader)
    return eval_loss, acc, f1


def test(model, device, tokenizer, args):
    test_data = DefectDiagnosisDataSet(tokenizer, args.max_len, args.data_dir, "test_defect_diagnosis",
                                       args.test_file_path)
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_defect_diagnosis)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    eval_loss = 0.0
    true_label = []
    pre_label = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            [loss, predict_label, _] = model.forward(input_ids, attention_mask, label)
            eval_loss += loss.item()
            true_label.extend(label.cpu().numpy())
            pre_label.extend(predict_label.cpu().numpy())
    true_label = np.array(true_label)
    pre_label = np.array(pre_label)
    f1 = f1_score(true_label, pre_label, average='micro')
    acc = accuracy_score(true_label, pre_label)
    eval_loss = eval_loss / len(test_data_loader)
    return eval_loss, acc, f1


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--train_file_path', default='task_data/train_defect_diagnosis.json', type=str, help='')
    parser.add_argument('--dev_file_path', default='task_data/dev_defect_diagnosis.json', type=str, help='')
    parser.add_argument('--test_file_path', default='task_data/test_defect_diagnosis.json', type=str, help='')
    parser.add_argument('--vocab_path', default="chinese_L-12_H-768_A-12/vocab.txt", type=str, help='')
#     parser.add_argument('--pretrained_model_path', default="chinese_L-12_H-768_A-12/", type=str, help='')
#     parser.add_argument('--pretrained_model_path', default="output_dir_ori_bert_only_mlm/checkpoint-295000/", type=str, help='')
    parser.add_argument('--pretrained_model_path', default="../output_dir_ori_bert_only_mlm_v2/checkpoint-295000/", type=str, help='')
    parser.add_argument('--data_dir', default='task_data/', type=str, help='')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='')
    parser.add_argument('--train_batch_size', default=16, type=int, help='')
    parser.add_argument('--test_batch_size', default=8, type=int, help='')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='')
    parser.add_argument('--save_model_steps', default=19, type=int, help='')
    parser.add_argument('--logging_steps', default=5, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='bp_output_dir/defect_diagnosis_output_dir', type=str,
                        help='')
    parser.add_argument('--test_model_path', default='bp_output_dir/defect_diagnosis_output_dir', type=str,
                        help='')
    parser.add_argument('--is_training', type=bool, default=True, help='')
    parser.add_argument('--is_testing', type=bool, default=False, help='')
    parser.add_argument('--seed', type=int, default=2020, help='')
    parser.add_argument('--max_len', type=int, default=256, help='')
    parser.add_argument('--num_labels', type=int, default=3, help='')
    return parser.parse_args()


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
#     if args.seed:
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         np.random.seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    
    for ipc in range(5, 12):
        defect_diagnosis_ori_path = "task_data/task23.xlsx"
        defect_diagnosis_path = "task_data/defect_diagnosis.json"
        get_defect_diagnosis(defect_diagnosis_ori_path, defect_diagnosis_path)

        train_defect_diagnosis_path = "task_data/train_defect_diagnosis.json"
        dev_defect_diagnosis_path = "task_data/dev_defect_diagnosis.json"
        test_defect_diagnosis_path = "task_data/test_defect_diagnosis.json"
        split_defect_diagnosis_data(defect_diagnosis_path, train_defect_diagnosis_path, dev_defect_diagnosis_path, test_defect_diagnosis_path)
        args.train_file_path = train_defect_diagnosis_path
        args.dev_file_path = dev_defect_diagnosis_path
        args.test_file_path = test_defect_diagnosis_path

        output_dit_list = ["ori_output_dir/defect_diagnosis_output_dir", "bp_output_dir/defect_diagnosis_output_dir", "po_output_dir/defect_diagnosis_output_dir"]
        for ip, pretrain_path in enumerate(["chinese_L-12_H-768_A-12/", "../output_dir_ori_bert_only_mlm_v2/checkpoint-295000/", "output_dir_ori_bert_only_mlm/checkpoint-295000/", ]):
            args.output_dir = output_dit_list[ip]
            args.pretrained_model_path = pretrain_path
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            if args.is_training:
                for irt in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    time.sleep(2)
                    model = DefectDiagnosisModel.from_pretrained(args.pretrained_model_path, num_labels=args.num_labels)
                    train(model, device, tokenizer, args, irt, ipc)
                    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
