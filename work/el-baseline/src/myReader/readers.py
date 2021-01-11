import json
import os
from os.path import join
import random

from loguru import logger
import paddle
from paddle import fluid
import numpy as np

from ernie.tokenizing_ernie import ErnieTokenizer
from myReader.nil_types import *
from myUtils.args import get_parser, VersionConfig

mention2id = None
id2entity = None


class ELReader:
    def __init__(self, tokenizer: ErnieTokenizer, args,
                 mode, version_cfg, shuffle=False, is_predict=False):
        assert mode in ['train', 'dev', 'test', 'test_b']
        assert tokenizer.pad_id == 0

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        self.is_predict = is_predict
        self.shuffle = shuffle
        self.version_cfg = version_cfg
        self.batch_size = self.args.per_device_train_batch_size if self.mode == 'train' else self.args.per_device_eval_batch_size
        self.max_seq_length = version_cfg.max_seq_length if 'test' in mode else args.max_seq_length

        self.samples = self._load_data()
        self.mention2id, self.id2entity = self._load_global_dict()

    def _load_data(self):
        """load {train}{dev}{test}.json"""
        logger.info(f'loading from {self.mode}.json...')
        with open(join(self.args.data_dir, f'{self.mode}.json'), encoding='utf8') as f:
            samples = [json.loads(line) for line in f.readlines()]
        return samples

    def _build_dict(self, mention2id_path, id2entity_path):
        mention2id = dict()
        id2entity = dict()
        with open(os.path.join(self.args.data_dir, 'kb.json'), 'r', encoding='utf-8') as fin:
            kb = [json.loads(line.strip()) for line in fin.readlines()]
            for i in range(len(kb)):
                id2entity[kb[i]['subject_id']] = kb[i]
                _mentions = [kb[i]['subject']]
                for _mention in _mentions:
                    if _mention in mention2id.keys():
                        mention2id[_mention].append(kb[i]['subject_id'])
                    else:
                        mention2id[_mention] = [kb[i]['subject_id']]
        for k, v in mention2id.items():
            mention2id[k] = list(set(v))
        with open(mention2id_path, 'w', encoding='utf-8') as fot:
            fot.write(json.dumps(mention2id, ensure_ascii=False))
        with open(id2entity_path, 'w', encoding='utf-8') as fot:
            fot.write(json.dumps(id2entity, ensure_ascii=False))

    def _load_global_dict(self):
        """由于这几个文件加载很慢，使用全局加载"""
        global mention2id
        global id2entity

        _mention2id_path = "./dicts/mention2id.json"
        _id2entity_path = "./dicts/id2entity.json"

        if mention2id is None:
            if not os.path.exists(_mention2id_path):
                if not os.path.exists('./dicts'):
                    os.mkdir('./dicts')
                self._build_dict(_mention2id_path, _id2entity_path)
            mention2id = json.load(open(_mention2id_path, encoding='utf8'))
            logger.info("load sub2id.json, id2entity.json, ......")
        if id2entity is None:
            id2entity = json.load(open(_id2entity_path, encoding='utf8'))
        return mention2id, id2entity

    def _get_nil_type_ids(self, mention_dict):
        '''训练集验证集中部分样例NIL标签有多个类别'''
        kb_id_label = mention_dict['kb_id']
        if kb_id_label.isdigit():
            entity_ = self.id2entity[kb_id_label]
            nil_type_name = 'NIL_' + entity_['type']
        else:
            nil_type_name = kb_id_label
        nil_type_id_list = []
        if '|' in nil_type_name:
            nil_type_names = nil_type_name.split('|')
            for n in nil_type_names:
                if not n.startswith('NIL_'):
                    n = 'NIL_' + n
                nil_type_id = TYPE2ID.get(n)
                if nil_type_id is not None:
                    nil_type_id_list.append(nil_type_id)
                    # 只有训练集才返回多个nil类别，dev只返回第一个
                    if self.mode != 'train':
                        break
        else:
            nil_type_id = TYPE2ID.get(nil_type_name)
            if nil_type_id is not None:
                nil_type_id_list.append(nil_type_id)

        if not nil_type_id_list:
            nil_type_id_list.append(TYPE2ID['NIL_Other'])
        return nil_type_id_list

    def sample_reader(self):
        random.seed(self.args.seed)
        if self.shuffle:
            random.shuffle(self.samples)

        def extract_entity_feature(kbid):
            if not kbid:
                return ""
            entity = self.id2entity[kbid]
            desc = {}
            for rel in entity['data']:
                desc[rel['predicate']] = rel['object']
            text_b = ','.join(list(desc.values()))
            return text_b

        for sample in self.samples:
            text_a = sample['text']
            for mention_dict in sample['mention_data']:
                mention_text = mention_dict['mention']
                mention_offset = int(mention_dict['offset'])
                kbids = self.mention2id.get(mention_text, [])
                if not kbids:
                    kbids.append(None)
                random.shuffle(kbids)

                # 随机负采样比例
                NEG_RATE = 2
                neg_count = 0
                for kbid in kbids:
                    text_b = extract_entity_feature(kbid)
                    ####
                    # 由 cls + text_a + sep + text_b + sep 组成
                    ####

                    text_a = text_a[:mention_offset] + '#' + mention_text + '#' + text_a[mention_offset + len(mention_text):]

                    if self.args.other_mention:
                        text_a += ','.join([x['mention'] for x in sample['mention_data']])

                    ret_id, ret_id_type = self.tokenizer.encode(
                        text=text_a,
                        pair=text_b,
                        truncate_to=self.max_seq_length
                    )
                    raw_sample = {
                            'text_id': sample['text_id'],
                            'text': sample['text'],
                            'mention': mention_dict['mention'],
                            'offset': mention_dict['offset'],
                            'kb_id': kbid
                    }
                    model_inputs = {
                        'ret_id': ret_id,
                        'ret_id_type': ret_id_type
                    }
                    if self.is_predict:
                        yield model_inputs, raw_sample
                    else:
                        label_sim = 1 if mention_dict['kb_id'] == kbid else -1
                        if not self.is_predict and label_sim == -1:
                            if neg_count == 2:
                                continue
                            else:
                                neg_count += 1
                        for label_nil in self._get_nil_type_ids(mention_dict):
                            yield model_inputs, raw_sample, label_sim, label_nil

    def batch_reader(self, ):
        def _batch():
            reader = fluid.io.batch(self.sample_reader, batch_size=self.batch_size)
            for batch in reader():
                batch.sort(key=lambda sample: len(sample[0]['ret_id']), reverse=True)
                batch_seq_len = len(batch[0][0]['ret_id'])

                ret_ids = [sample[0]['ret_id'] for sample in batch]
                ret_id_types = [sample[0]['ret_id_type'] for sample in batch]
                lengths = [len(sample[0]['ret_id']) for sample in batch]

                for i in range(1, len(ret_ids)):
                    pad_len = batch_seq_len - lengths[i]
                    ret_ids[i] = np.append(ret_ids[i], np.zeros(pad_len, dtype=np.int))
                    ret_id_types[i] = np.append(ret_id_types[i], np.ones(pad_len, dtype=np.int))

                model_inputs = {
                    'src_ids': np.stack(ret_ids),
                    'sent_ids': np.stack(ret_id_types)
                }

                raw_samples = [sample[1] for sample in batch]
                if self.is_predict:
                    yield model_inputs, raw_samples
                else:
                    labels_sim = np.array([sample[2] for sample in batch], 'float32')  # tanh 1, -1
                    labels_nil = np.array([sample[3] for sample in batch], 'int64')  # tanh 1, -1
                    yield model_inputs, raw_samples, labels_sim, labels_nil
        buffered_reader = paddle.reader.buffered(_batch, self.batch_size * 2)
        return buffered_reader()


if __name__ == '__main__':
    args = get_parser()
    version_cfg = VersionConfig(max_seq_length=128)
    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
    dataset = ELReader(tokenizer, args, 'train', version_cfg=version_cfg, is_predict=True)
    for batch in dataset.batch_reader():
        print(batch)