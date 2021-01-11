import json
from os.path import join
import paddle.fluid.layers as L
import paddle.fluid as F
import paddle.fluid.dygraph as D
from ernie.modeling_ernie import ErnieModel
from myReader.nil_types import *


class ModelWithErnie(D.Layer):
    # 减小计算量
    def __init__(self, args, version_cfg):
        super(ModelWithErnie, self).__init__()
        self.args = args
        self.version_cfg = version_cfg
        self.ernie = ErnieModel.from_pretrained(args.model_name_or_path, num_out_pooler=1)

        self.entity_embed_layer = D.Embedding([len(TYPE2ID), 768])
        # self.word_embed_layer = self.ernie_mt.word_emb

        self.sim_ffn = D.Linear(768, 1)  # sim
        self.nil_ffn = D.Linear(768, len(TYPE2ID))

    def forward(self, inputs: dict):
        pooled, _, = self.ernie(**inputs)
        out0 = L.tanh(L.squeeze(self.sim_ffn(pooled), [1]))
        out1 = L.softmax(self.nil_ffn(pooled))
        return out0, out1
