import json


class FusionTool:
    """ 用于模型融合或者单个模型debug预测分值信息，每条sample对应一个对象 """

    def __init__(self, text_id, text, thresh=-0.8):
        """

        Args:
            text_id:
            text:
            num_model: deprecated
            thresh: value is -0.8
        """
        self.text_id = text_id
        self.text = text
        self.num_model = 1
        self.thresh = thresh
        self.mention_data_sim = {}  # k_mention : v_pred |sim
        self.mention_data_nil = {}  # k_mention : v_pred |nil

        '''
        NIL:
        {
            '天下没有不散的宴席[||]0': {
                'NIL_Medicine':{'nil_type': 'NIL_Medicine', 'offset': '0', 'mention': '天下没有不散的宴席', 'score': 5.6911836e-06}, 
                'NIL_Brand': {'nil_type': 'NIL_Brand', 'offset': '0', 'mention': '天下没有不散的宴席', 'score': 7.9810634e-05}, 
                'NIL_Website': {'nil_type': 'NIL_Website', 'offset': '0', 'mention': '天下没有不散的宴席', 'score': 8.699003e-05}
            }
        }
        SIM:
        {
            '天下没有不散的宴席[||]0': {
                '28270': {'kb_id': '28270', 'offset': '0', 'mention': '天下没有不散的宴席', 'score': -0.7194435596466064}
            }
        }
        '''
        
    def __add__(self, other):
        raise NotImplementedError

    def merge(self, other):
        """
        融合运算符, 对每个模型预测分值取平均
        """

        assert other.text_id == self.text_id
        assert len(other.mention_data_nil) == len(self.mention_data_nil)
        assert len(other.mention_data_sim) == len(self.mention_data_sim)

        self.num_model += 1
        for k_sim, o_v in other.mention_data_sim.items():
            assert len(self.mention_data_sim[k_sim]) == len(o_v)
            for kbid, o_v_pred in o_v.items():
                self.mention_data_sim[k_sim][kbid]['score'] += o_v_pred['score']

        for k_nil, o_v in other.mention_data_nil.items():
            assert len(self.mention_data_nil[k_nil]) == len(o_v)
            for nil_type, o_v_pred in o_v.items():
                self.mention_data_nil[k_nil][nil_type] += o_v_pred['score']


    def add_predict_sim(self, mention: str, offset: str, kb_id: str, score: float):
        k_mention = mention + '[||]' + offset
        v_pred = {
            "kb_id": kb_id,
            "offset": offset,
            "mention": mention,
            "score": score
        }

        if k_mention not in self.mention_data_sim:
            self.mention_data_sim[k_mention] = {
                kb_id: v_pred
            }
        elif kb_id not in self.mention_data_sim[k_mention]:
            self.mention_data_sim[k_mention][kb_id] = v_pred
        else:
            self.mention_data_sim[k_mention][kb_id]["score"] += score

    def add_predict_nil(self, mention: str, offset: str, nil_type: str, score: float):
        k_mention = mention + '[||]' + offset
        v_pred = {'nil_type': nil_type, 'offset': offset, 'mention': mention, 'score': score}

        if k_mention not in self.mention_data_nil:
            self.mention_data_nil[k_mention] = {
                nil_type: v_pred
            }
        elif nil_type not in self.mention_data_nil[k_mention]:
            self.mention_data_nil[k_mention][nil_type] = v_pred
        else:
            self.mention_data_nil[k_mention][nil_type]["score"] += score

    def _argmax_kbid(self, k_m, debug=False):
        """ return: (kb_id, is_in_kb) type(str, bool)"""
        if k_m not in self.mention_data_sim:
            return [], False
        st = sorted(self.mention_data_sim[k_m].values(), key=lambda d: d['score'], reverse=True)
        if debug:
            return st[:3], True

        kb_id = st[0]['kb_id']
        score = st[0]['score'] / self.num_model
        if score < self.thresh:
            return kb_id, False
        return kb_id, True

    def _argmax_niltype(self, k_m, debug=False):
        """ return most like nil type """
        st = sorted(self.mention_data_nil[k_m].values(), key=lambda d: d['score'], reverse=True)
        if debug:
            return st[:3]
        return st[0]['nil_type']

    def str_for_test(self):
        out_str = {
            'text_id': self.text_id,
            'text': self.text,
            'mention_data': []
        }
        for k_mention, v_pred_list in self.mention_data_nil.items():  # only nil content is complete
            mention, offset = k_mention.split('[||]')
            out_m = {
                'kb_id': None,
                'mention': mention,
                'offset': offset
            }
            kb_id, isInKB = self._argmax_kbid(k_mention)
            if isInKB:
                out_m['kb_id'] = kb_id
            else:
                out_m['kb_id'] = self._argmax_niltype(k_mention)
            out_str['mention_data'].append(out_m)
        return json.dumps(out_str, ensure_ascii=False)

    def __str__(self):
        return self.str_for_test()

    def str_for_debug(self):
        '''sim , nil, 返回已经添加的所有version加和后概率最大的前三个'''
        out_str = {
            'text_id': self.text_id,
            'text': self.text,
            'mention_data_sim': {},
            'mention_data_nil': {}
        }
        for k_mention, v_pred_list in self.mention_data_nil.items():
            kb_ids_dict, _ = self._argmax_kbid(k_mention, debug=True)
            sims = [(k['kb_id'], float(k['score'])) for k in kb_ids_dict]
            out_str['mention_data_sim'][k_mention] = sims

            nil_dict = self._argmax_niltype(k_mention, debug=True)
            nils = [(k['nil_type'], float(k['score'])) for k in nil_dict]
            out_str['mention_data_nil'][k_mention] = nils

        return json.dumps(out_str, ensure_ascii=False)


if __name__ == '__main__':
    import pickle
    ans_path = '../../output/ans_v5_dev.txt'
    obj = pickle.loads(open(ans_path, 'rb').read())
    print(obj.mention_data_sim)
    print(obj.mention_data_nil)