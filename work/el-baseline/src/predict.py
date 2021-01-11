""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
import json
from collections import defaultdict
from os.path import join
import os
import paddle.fluid as F
import paddle.fluid.dygraph as D
from loguru import logger
import numpy as np
from tqdm import tqdm

from myModel.models import ModelWithErnie
from myReader.readers import ELReader
from myReader.nil_types import *
from myUtils.args import get_parser, VersionConfig
from myUtils.fusion import FusionTool
from ernie.tokenizing_ernie import ErnieTokenizer
import pickle

TASK_DICT = {
    'sim': 0,
    'nil': 1,
}


args = get_parser()
DEVICE = F.CPUPlace() if args.no_cuda else F.CUDAPlace(0)
logger.info(f'use {DEVICE}!')


def predict_combine(model, ans, data_loader: ELReader):
    with D.guard(DEVICE):
        with tqdm() as t:
            t.set_description("test-combine")
            for model_inputs, raw_samples in data_loader.batch_reader():
                for k, v in model_inputs.items():
                    if isinstance(v, np.ndarray):
                        model_inputs[k] = D.to_variable(v)
                out_sim, out_nil = model(model_inputs)
                assert len(out_sim) == out_nil.shape[0]
                for i in range(len(out_sim)):
                    text_id = raw_samples[i]['text_id']
                    text = raw_samples[i]['text']
                    mention = raw_samples[i]['mention']
                    kb_id = raw_samples[i]['kb_id']
                    offset = raw_samples[i]['offset']

                    # add sim result
                    if text_id not in ans:
                        ans[text_id] = FusionTool(text_id, text)
                    score_sim = float(out_sim.numpy()[i])
                    if kb_id is not None:
                        # 为none时表示该mention在知识库中没有召回
                        ans[text_id].add_predict_sim(mention, offset, kb_id, score_sim)

                    # add nil result
                    for j in range(out_nil.shape[1]):
                        type_name = ID2TYPE[j]
                        ans[text_id].add_predict_nil(mention, offset, type_name, out_nil[i][j].numpy()[0])
                t.update(len(out_sim))
    return ans


def main(mode='dev', threshhold=-0.6):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info("Training/evaluation parameters %s" % args)
    assert mode in ['dev', 'test', 'train', 'test_b']
    # from pretrained bert load token and model
    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)

    ans = {}
    model_dir = args.load_from
    with D.guard(DEVICE):
        version_cfg = VersionConfig()
        version_cfg.load(model_dir)
        model = ModelWithErnie(args, version_cfg)
        logger.info(f'load from {model_dir}')
        model_stat, _ = D.load_dygraph(join(model_dir, "weights"))
        assert model_stat is not None
        model.set_dict(model_stat)
        model.eval()
    data_loader = ELReader(tokenizer, args, mode, version_cfg, is_predict=True)
    ans = predict_combine(model, ans, data_loader)

    # output every version for debug
    text_ids = list(ans.keys())
    text_ids.sort(key=lambda x: int(x))

    # save submit file
    with open(join(model_dir, f'result_{mode}.json'), 'w', encoding='utf8') as f:
        for text_id in text_ids:
            ans[text_id].thresh = threshhold
            f.write(ans[text_id].str_for_test()+'\n')


if __name__ == '__main__':
    mode = args.mode
    assert 'train' in mode or 'dev' in mode or 'test' in mode
    main(mode, 0)
