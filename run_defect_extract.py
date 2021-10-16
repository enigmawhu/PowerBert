# -*- coding:utf-8 -*-
# @project: PowerBert
# @filename: run_defect_extract
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/5 20:29
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
from task_data_set import DefectExtractDataSet, collate_func_defect_extract
from model import DefectExtractModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import re
from sklearn.metrics import classification_report, f1_score, accuracy_score
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
    train_data = DefectExtractDataSet(tokenizer, args.max_len, args.data_dir, "train_defect_extract",
                                      args.label1_dict_path, args.label2_dict_path, args.label3_dict_path,
                                      args.label4_dict_path, args.train_file_path)
    dev_data = DefectExtractDataSet(tokenizer, args.max_len, args.data_dir, "dev_defect_extract",
                                    args.label1_dict_path, args.label2_dict_path, args.label3_dict_path,
                                    args.label4_dict_path, args.dev_file_path)
    test_data = DefectExtractDataSet(tokenizer, args.max_len, args.data_dir, "test_defect_extract",
                                     args.label1_dict_path, args.label2_dict_path, args.label3_dict_path,
                                     args.label4_dict_path, args.test_file_path)

    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func_defect_extract)
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
            label1 = batch["label1"].to(device)
            label2 = batch["label2"].to(device)
            label3 = batch["label3"].to(device)
            label4 = batch["label4"].to(device)
            label5_start = batch["label5_start"].to(device)
            label5_end = batch["label5_end"].to(device)
            outputs = model.forward(input_ids, attention_mask, label1, label2, label3, label4, label5_start, label5_end)
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


        eval_loss, eval_t_score, eval_label1_f1, eval_label1_acc, eval_label2_f1, eval_label2_acc, eval_label3_f1, eval_label3_acc, eval_label4_f1, eval_label4_acc, eval_label5_f1, eval_label5_em = evaluate(
            model, device, dev_data, args)
        logger.info(
            "eval_loss is {}, eval_t_score is {}, eval_label1_f1 is {}, eval_label1_acc is {}, eval_label2_f1 is {}, eval_label2_acc is {}, eval_label3_f1 is {}, eval_label3_acc is {}, eval_label4_f1 is {}, eval_label4_acc is {}, eval_label5_f1 is {}, eval_label5_em is {}".format(
                eval_loss, eval_t_score, eval_label1_f1, eval_label1_acc, eval_label2_f1, eval_label2_acc,
                eval_label3_f1, eval_label3_acc, eval_label4_f1, eval_label4_acc, eval_label5_f1,
                eval_label5_em))

        test_loss, test_t_score, test_label1_f1, test_label1_acc, test_label2_f1, test_label2_acc, test_label3_f1, test_label3_acc, test_label4_f1, test_label4_acc, test_label5_f1, test_label5_em = evaluate(
            model, device, test_data, args)
        logger.info(
            "test_loss is {}, test_t_score is {}, test_label1_f1 is {}, test_label1_acc is {}, test_label2_f1 is {}, test_label2_acc is {}, test_label3_f1 is {}, test_label3_acc is {}, test_label4_f1 is {}, test_label4_acc is {}, test_label5_f1 is {}, test_label5_em is {}".format(
                test_loss, test_t_score, test_label1_f1, test_label1_acc, test_label2_f1, test_label2_acc,
                test_label3_f1, test_label3_acc, test_label4_f1, test_label4_acc, test_label5_f1,
                test_label5_em))

        if eval_t_score >= max_acc:
            max_acc = eval_t_score
            output_dir = os.path.join(args.output_dir, "checkpoint")
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            json_output_dir = os.path.join(output_dir, "json_data.json")
            fin = open(json_output_dir, "w", encoding="utf-8")
            fin.write(json.dumps(
                {"eval_t_score": float(eval_t_score), "eval_label1_f1": float(eval_label1_f1),
                 "eval_label1_acc": float(eval_label1_acc), "eval_label2_f1": float(eval_label2_f1),
                 "eval_label2_acc": float(eval_label2_acc), "eval_label3_f1": float(eval_label3_f1),
                 "eval_label3_acc": float(eval_label3_acc), "eval_label4_f1": float(eval_label4_f1),
                 "eval_label4_acc": float(eval_label4_acc), "eval_label5_f1": float(eval_label5_f1),
                 "eval_label5_em": float(eval_label5_em), "test_t_score": float(test_t_score),
                 "test_label1_f1": float(test_label1_f1), "test_label1_acc": float(test_label1_acc),
                 "test_label2_f1": float(test_label2_f1), "test_label2_acc": float(test_label2_acc),
                 "test_label3_f1": float(test_label3_f1), "test_label3_acc": float(test_label3_acc),
                 "test_label4_f1": float(test_label4_f1), "test_label4_acc": float(test_label4_acc),
                 "test_label5_f1": float(test_label5_f1), "test_label5_em": float(test_label5_em)},
                ensure_ascii=False, indent=4))
            fin.close()
        model.train()


def evaluate(model, device, dev_data, args):
    test_sampler = SequentialSampler(dev_data)
    test_data_loader = DataLoader(dev_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_defect_extract)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    eval_loss = 0.0
    true_label1, true_label2, true_label3, true_label4 = [], [], [], []
    pre_label1, pre_label2, pre_label3, pre_label4 = [], [], [], []
    true_answer_list = []
    pre_answer_list = []
    label5_f1_list = []
    label5_em_list = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label1 = batch["label1"].to(device)
            label2 = batch["label2"].to(device)
            label3 = batch["label3"].to(device)
            label4 = batch["label4"].to(device)
            label5_start = batch["label5_start"].to(device)
            label5_end = batch["label5_end"].to(device)
            outputs = model.forward(input_ids, attention_mask, label1, label2, label3, label4, label5_start, label5_end)
            eval_loss += outputs[0].item()
            true_label1.extend(label1.cpu().numpy())
            pre_label1.extend(outputs[6].cpu().numpy())
            true_label2.extend(label2.cpu().numpy())
            pre_label2.extend(outputs[8].cpu().numpy())
            true_label3.extend(label3.cpu().numpy())
            pre_label3.extend(outputs[10].cpu().numpy())
            true_label4.extend(label4.cpu().numpy())
            pre_label4.extend(outputs[12].cpu().numpy())

            for tokens, y_s, y_e, t_y_s, t_y_e in zip(batch["tokens"], outputs[14].cpu().numpy(),
                                                      outputs[16].cpu().numpy(), label5_start.cpu().numpy(),
                                                      label5_end.cpu().numpy()):
                best_answer, true_answer = find_best_answer(tokens, y_s, y_e, t_y_s, t_y_e)
                true_answer_list.append(true_answer)
                pre_answer_list.append(best_answer)
                f1 = calc_f1_score([true_answer], best_answer)
                em = calc_em_score([true_answer], best_answer)
                label5_f1_list.append(f1)
                label5_em_list.append(em)

    label1_f1 = f1_score(np.array(true_label1), np.array(pre_label1), average='micro')
    label1_acc = accuracy_score(np.array(true_label1), np.array(pre_label1))
    label2_f1 = f1_score(np.array(true_label2), np.array(pre_label2), average='micro')
    label2_acc = accuracy_score(np.array(true_label2), np.array(pre_label2))
    label3_f1 = f1_score(np.array(true_label3), np.array(pre_label3), average='micro')
    label3_acc = accuracy_score(np.array(true_label3), np.array(pre_label3))
    label4_f1 = f1_score(np.array(true_label4), np.array(pre_label4), average='micro')
    label4_acc = accuracy_score(np.array(true_label4), np.array(pre_label4))
    label5_f1 = np.mean(np.array(label5_f1_list))
    label5_em = np.mean(np.array(label5_em_list))
    total_score = []
    for t_l1, p_l1, t_l2, p_l2, t_l3, p_l3, t_l4, p_l4, f1_5 in zip(true_label1, pre_label1, true_label2,
                                                                    pre_label2, true_label3, pre_label3,
                                                                    true_label4, pre_label4, label5_f1_list):
        if t_l1 == p_l1 and t_l2 == p_l2 and t_l3 == p_l3 and t_l4 == p_l4:
            total_score.append(1.0)
        else:
            total_score.append(0.0)
    t_score = np.mean(np.array(total_score))
    eval_loss = eval_loss / len(test_data_loader)
    return eval_loss, t_score, label1_f1, label1_acc, label2_f1, label2_acc, label3_f1, label3_acc, label4_f1, label4_acc, label5_f1, label5_em


def find_best_answer(tokens, y_s, y_e, t_y_s, t_y_e):
    true_answer = "".join(tokens[t_y_s: t_y_e]).replace("[unused1]", " ")
    if y_s > y_e:
        best_answer = ""
    else:
        best_answer = "".join(tokens[y_s:y_e]).replace("[unused1]", " ")
    return best_answer, true_answer


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = list(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = list(temp_str)
        segs_out.extend(ss)

    return segs_out


def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def test(model, device, tokenizer, args):   
    model.to(device)
    test_data = DefectExtractDataSet(tokenizer, args.max_len, args.data_dir, "test_defect_extract",
                                     args.label1_dict_path, args.label2_dict_path, args.label3_dict_path,
                                     args.label4_dict_path, args.test_file_path)
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_defect_extract)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    eval_loss = 0.0
    true_label1, true_label2, true_label3, true_label4 = [], [], [], []
    pre_label1, pre_label2, pre_label3, pre_label4 = [], [], [], []
    true_answer_list = []
    pre_answer_list = []
    label5_f1_list = []
    label5_em_list = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label1 = batch["label1"].to(device)
            label2 = batch["label2"].to(device)
            label3 = batch["label3"].to(device)
            label4 = batch["label4"].to(device)
            label5_start = batch["label5_start"].to(device)
            label5_end = batch["label5_end"].to(device)
            outputs = model.forward(input_ids, attention_mask, label1, label2, label3, label4, label5_start, label5_end)
            eval_loss += outputs[0].item()
            true_label1.extend(label1.cpu().numpy())
            pre_label1.extend(outputs[6].cpu().numpy())
            true_label2.extend(label2.cpu().numpy())
            pre_label2.extend(outputs[8].cpu().numpy())
            true_label3.extend(label3.cpu().numpy())
            pre_label3.extend(outputs[10].cpu().numpy())
            true_label4.extend(label4.cpu().numpy())
            pre_label4.extend(outputs[12].cpu().numpy())

            for tokens, y_s, y_e, t_y_s, t_y_e in zip(batch["tokens"], outputs[14].cpu().numpy(),
                                                      outputs[16].cpu().numpy(), label5_start.cpu().numpy(),
                                                      label5_end.cpu().numpy()):
                best_answer, true_answer = find_best_answer(tokens, y_s, y_e, t_y_s, t_y_e)
                true_answer_list.append(true_answer)
                pre_answer_list.append(best_answer)
                f1 = calc_f1_score([true_answer], best_answer)
                em = calc_em_score([true_answer], best_answer)
                label5_f1_list.append(f1)
                label5_em_list.append(em)

    label1_f1 = f1_score(np.array(true_label1), np.array(pre_label1), average='micro')
    label1_acc = accuracy_score(np.array(true_label1), np.array(pre_label1))
    label2_f1 = f1_score(np.array(true_label2), np.array(pre_label2), average='micro')
    label2_acc = accuracy_score(np.array(true_label2), np.array(pre_label2))
    label3_f1 = f1_score(np.array(true_label3), np.array(pre_label3), average='micro')
    label3_acc = accuracy_score(np.array(true_label3), np.array(pre_label3))
    label4_f1 = f1_score(np.array(true_label4), np.array(pre_label4), average='micro')
    label4_acc = accuracy_score(np.array(true_label4), np.array(pre_label4))
    label5_f1 = np.mean(np.array(label5_f1_list))
    label5_em = np.mean(np.array(label5_em_list))
    total_score = []
    for t_l1, p_l1, t_l2, p_l2, t_l3, p_l3, t_l4, p_l4, f1_5 in zip(true_label1, pre_label1, true_label2,
                                                                    pre_label2, true_label3, pre_label3,
                                                                    true_label4, pre_label4, label5_f1_list):
        if t_l1 == p_l1 and t_l2 == p_l2 and t_l3 == p_l3 and t_l4 == p_l4:
            total_score.append(1.0)
        else:
            total_score.append(0.0)
    t_score = np.mean(np.array(total_score))
    eval_loss = eval_loss / len(test_data_loader)
    return eval_loss, t_score, label1_f1, label1_acc, label2_f1, label2_acc, label3_f1, label3_acc, label4_f1, label4_acc, label5_f1, label5_em


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', type=str, help='')
    parser.add_argument('--train_file_path', default='task_data/train_defect_extract.json', type=str, help='')
    parser.add_argument('--dev_file_path', default='task_data/dev_defect_extract.json', type=str, help='')
    parser.add_argument('--test_file_path', default='task_data/test_defect_extract.json', type=str, help='')
    parser.add_argument('--vocab_path', default="chinese_L-12_H-768_A-12/vocab.txt", type=str, help='')
    parser.add_argument('--label1_dict_path', default="task_data/defect_extract_label1.txt", type=str, help='')
    parser.add_argument('--label2_dict_path', default="task_data/defect_extract_label2.txt", type=str, help='')
    parser.add_argument('--label3_dict_path', default="task_data/defect_extract_label3.txt", type=str, help='')
    parser.add_argument('--label4_dict_path', default="task_data/defect_extract_label4.txt", type=str, help='')
#     parser.add_argument('--pretrained_model_path', default="chinese_L-12_H-768_A-12/", type=str, help='')
#     parser.add_argument('--pretrained_model_path', default="output_dir_ori_bert_only_mlm/checkpoint-295000/", type=str, help='')
    parser.add_argument('--pretrained_model_path', default="../output_dir_ori_bert_only_mlm_v2/checkpoint-295000/", type=str, help='')
    parser.add_argument('--data_dir', default='task_data/', type=str, help='')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='')
    parser.add_argument('--train_batch_size', default=16, type=int, help='')
    parser.add_argument('--test_batch_size', default=4, type=int, help='')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='')
    parser.add_argument('--save_model_steps', default=9, type=int, help='')
    parser.add_argument('--logging_steps', default=5, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='bp_output_dir/defect_extract_output_dir', type=str,
                        help='')
    parser.add_argument('--test_model_path', default='bp_output_dir/defect_extract_output_dir', type=str,
                        help='')
    parser.add_argument('--is_training', type=bool, default=True, help='')
    parser.add_argument('--is_testing', type=bool, default=False, help='')
    parser.add_argument('--seed', type=int, default=2020, help='')
    parser.add_argument('--max_len', type=int, default=256, help='')
    parser.add_argument('--num_label1', type=int, default=27, help='')
    parser.add_argument('--num_label2', type=int, default=10, help='')
    parser.add_argument('--num_label3', type=int, default=14, help='')
    parser.add_argument('--num_label4', type=int, default=23, help='')
    parser.add_argument('--num_label5', type=int, default=2, help='')
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
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    if args.is_training:
        output_dit_list = ["ori_output_dir/defect_extract_output_dir", "bp_output_dir/defect_extract_output_dir", "po_output_dir/defect_extract_output_dir"]
        for ip, pretrain_path in enumerate(["chinese_L-12_H-768_A-12/", "../output_dir_ori_bert_only_mlm_v2/checkpoint-295000/", "output_dir_ori_bert_only_mlm/checkpoint-295000/", ]):
            args.output_dir = output_dit_list[ip]
            args.pretrained_model_path = pretrain_path
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            if args.is_training:
                model = DefectExtractModel.from_pretrained(args.pretrained_model_path, num_label1=args.num_label1,
                                               num_label2=args.num_label2, num_label3=args.num_label3,
                                               num_label4=args.num_label4, num_label5=args.num_label5)
                train(model, device, tokenizer, args)
                torch.cuda.empty_cache()

#     if args.is_testing:
#         model = DefectExtractModel.from_pretrained(args.test_model_path, num_label1=args.num_label1,
#                                                    num_label2=args.num_label2, num_label3=args.num_label3,
#                                                    num_label4=args.num_label4, num_label5=args.num_label5)
#         test_loss, test_t_score, test_label1_f1, test_label1_acc, test_label2_f1, test_label2_acc, test_label3_f1, test_label3_acc, test_label4_f1, test_label4_acc, test_label5_f1, test_label5_em = test(
#             model, device, tokenizer, args)
#         print(
#             "test_loss is {}, test_t_score is {}, test_label1_f1 is {}, test_label1_acc is {}, test_label2_f1 is {}, test_label2_acc is {}, test_label3_f1 is {}, test_label3_acc is {}, test_label4_f1 is {}, test_label4_acc is {}, test_label5_f1 is {}, test_label5_em is {}".format(
#                 test_loss, test_t_score, test_label1_f1, test_label1_acc, test_label2_f1, test_label2_acc,
#                 test_label3_f1, test_label3_acc, test_label4_f1, test_label4_acc, test_label5_f1,
#                 test_label5_em))


if __name__ == '__main__':
    main()
