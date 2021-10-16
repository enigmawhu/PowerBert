# -*- coding:utf-8 -*-
# @project: PowerBert
# @filename: run_pre_train_only_mlm
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/8/22 16:42
"""
    文件说明:
            
"""
import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers import BertTokenizer
from data_set import BertOriPreTrainDataSetV2, collate_func
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, tokenizer, args):
    """
    """
    tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_data = BertOriPreTrainDataSetV2(tokenizer=tokenizer, max_len=args.max_len, data_dir=args.data_dir,
                                        data_set_name="pre_train_only_mlm", only_mlm=args.only_mlm,
                                        mask_source_words=args.mask_source_words, mlm_probability=args.mlm_probability,
                                        max_pred=args.max_pred, skipgram_prb=args.skipgram_prb,
                                        skipgram_size=args.skipgram_size, path_file=args.file_path,
                                        is_overwrite=args.is_overwrite)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size,
                                   collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)

    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    device_ids = list(range(torch.cuda.device_count()))
    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # 判断是否使用混合精度训练
    if args.fp16:
        # 如果使用，导入apex包，并且将原始model和optimizer按照混合精度等级进行重新初始化
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2], output_device=0)

        # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_loader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)


    # 清空cuda缓存
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    model.train()
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 开始训练模型
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            segment_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            # print(input_ids)
            # print(labels)
            outputs = model.forward(input_ids=input_ids, token_type_ids=segment_ids, labels=labels)
            loss = outputs[0]
            #print(loss)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            #print(loss)
            #print(loss.item())
            tr_loss += loss.item()
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
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
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()
    # 全部训练完成，保存最后一个模型
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--file_path', default='data_dir/all_data.json', type=str, help='预训练数据')
    parser.add_argument('--pretrained_model_path', default='chinese_L-12_H-768_A-12', type=str,
                        help='预训练的Roberta模型的路径')
    parser.add_argument('--data_dir', default='data_dir', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=72, type=int, help='训练时每个batch的大小')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--save_model_steps', default=5000, type=int, help='保存训练模型步数')
    parser.add_argument('--logging_steps', default=10, type=int, help='保存训练日志的步数')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--mlm_probability', default=0.15, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir_ori_bert_only_mlm_v2/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度')
    parser.add_argument("--fp16", type=bool, default=False, help="是否开启fp16混合精度训练")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="混合训练等级['O0', 'O1', 'O2', 'O3']，一般为O1")
    parser.add_argument('--only_mlm', default=True, type=bool, help='')
    parser.add_argument('--mask_source_words', default=True, type=bool, help='')
    parser.add_argument('--no_cuda', default=False, type=bool, help='')
    parser.add_argument('--is_overwrite', default=False, type=bool, help='')
    parser.add_argument('--max_pred', default=12, type=int, help='')
    parser.add_argument('--skipgram_prb', default=0.0, type=float, help='')
    parser.add_argument('--skipgram_size', default=1, type=int, help='')
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--n_gpu", type=int, default=4, help="")

    return parser.parse_args()


def main():
    # 设置模型训练参数
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    print("torch.cuda.device_count():", torch.cuda.device_count())
    # args.device = device
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model_config = BertConfig.from_pretrained(args.pretrained_model_path)
    if args.pretrained_model_path:
        model = BertForMaskedLM.from_pretrained(args.pretrained_model_path)
    else:
        model = BertForMaskedLM(config=model_config)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=True)
    if args.local_rank == 0:
        torch.distributed.barrier()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # 开始训练
    train(model, device, tokenizer, args)


if __name__ == '__main__':
    main()
