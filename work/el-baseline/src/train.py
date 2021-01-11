import json
import os
from os.path import join
import paddle.fluid as F
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L
from ernie.optimization import AdamW, linear_warmup_decay
from loguru import logger
from visualdl import LogWriter
import numpy as np
from tqdm import tqdm

# gpu = None
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.optimization import Dygraph_linear_warmup
from myReader.readers import ELReader
from myUtils.utils import CountSmooth, strftime
from myUtils.args import get_parser, VersionConfig
from myModel.models import ModelWithErnie


args = get_parser()
VERSION_CONFIG = VersionConfig(
    sim_loss_func='mse',
    max_seq_length=args.max_seq_length,
)


def get_sim_loss_fun(scale: float = 1.0):
    if VERSION_CONFIG.sim_loss_func in [None, 'mse']:
        def mse(outputs, labels):
            return F.layers.mean(L.square(outputs - labels)) / scale  # 使用更小的loss, 从而减小梯度

        return mse
    elif VERSION_CONFIG.sim_loss_func == 'hinge':
        def hinge(outputs, labels):
            return F.layers.mean(1 - labels * outputs)
        return hinge
    else:
        raise NotImplementedError


def main():
    writer = LogWriter(join(args.log_dir, strftime()))
    device = F.CPUPlace() if args.no_cuda else F.CUDAPlace(0)

    logger.info(f'use {device}!')
    logger.info("Training/evaluation parameters %s" % args)

    # from pretrained ernie load token and model
    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)

    # initialize model and optimizer
    with D.guard(device):
        model = ModelWithErnie(args, VERSION_CONFIG)
        clip = F.clip.GradientClipByValue(min=-0.5, max=0.5)
        optimizer = AdamW(
            learning_rate=Dygraph_linear_warmup(
                learning_rate=args.learning_rate,
                decay_steps=args.max_steps * 2,
                end_learning_rate=args.learning_rate*0.01,
                warmup_steps=args.warmup_steps
            ),
            parameter_list=model.parameters(),
            grad_clip=clip,
            weight_decay=args.weight_decay
        )


    train_loader = ELReader(tokenizer=tokenizer, args=args,
                            mode='train', version_cfg=VERSION_CONFIG, shuffle=True, is_predict=False)
    dev_loader = ELReader(tokenizer=tokenizer, args=args,
                          mode='dev', version_cfg=VERSION_CONFIG, shuffle=True, is_predict=False)

    def forward_step(inputs, label_sim, label_nil, is_train=True):
        # loss缩放
        sim_scale = 1
        sim_loss_func = get_sim_loss_fun(sim_scale)

        with D.guard(device):
            # inputs to tensors
            for k, v in inputs.items():
                if isinstance(v, np.ndarray):
                    inputs[k] = D.to_variable(v)
            label_nil = D.to_variable(label_nil)
            label_sim = D.to_variable(label_sim)
            out_sim, out_nil = model(inputs)
            tp_sim = L.cast(L.cast(out_sim > 0, 'float32') == L.cast(label_sim == 1, 'float32'),
                            'float32').numpy().sum()  # (bs,)
            tp_nil = L.cast(L.argmax(out_nil, axis=-1) == label_nil, 'float32').numpy().sum()
            acc_sim = tp_sim / label_sim.shape[0]
            acc_nil = tp_nil / label_nil.shape[0]

            l1_sim = sim_loss_func(out_sim, label_sim)
            l1_nil = F.layers.mean(L.cross_entropy(out_nil, label_nil))
            loss = l1_sim + l1_nil

            if is_train:
                loss.backward()
                optimizer.minimize(loss)
                model.clear_gradients()

            if np.isnan(loss.numpy()):
                print(f'loss:{loss}, model get nan, check please!', inputs)
                raise ValueError

            step_info = {
                'loss_combine': round(loss.numpy()[0], 3),
                'loss_sim': round(l1_sim.numpy()[0] * sim_scale, 3),
                'loss_nil': round(l1_nil.numpy()[0], 3),
                'acc_sim': round(acc_sim, 2),
                'acc_nil': round(acc_nil, 2),
                'tp_sim': tp_sim,
                'tp_nil': tp_nil
            }

            return step_info

    def eval_model(max_samples=500):
        # 每个epoch检验
        eval_info = {
            'eval_num': max_samples,
            'dev_sim_loss': 0,
            'dev_nil_loss': 0,
            'dev_sim_acc': 0,
            'dev_nil_acc': 0
        }
        real_eval_num, sample_num = 0, 0
        model.eval()
        with D.guard(device):
            '''eval sim task'''
            logger.info("eval dev sim dataset...")

            for inputs, _, labels_sim, labels_nil in dev_loader.batch_reader():
                step_info = forward_step(inputs, labels_sim, labels_nil, is_train=False)
                real_eval_num += 1
                sample_num += labels_sim.shape[0]
                eval_info['dev_sim_loss'] += step_info['loss_sim']
                eval_info['dev_nil_loss'] += step_info['loss_nil']
                eval_info['dev_sim_acc'] += step_info['tp_sim']
                eval_info['dev_nil_acc'] += step_info['tp_nil']
                if real_eval_num > max_samples:
                    break

            eval_info['dev_sim_loss'] /= real_eval_num
            eval_info['dev_nil_loss'] /= real_eval_num
            eval_info['dev_sim_acc'] /= sample_num
            eval_info['dev_nil_acc'] /= sample_num
        model.train()
        return eval_info

    def train():
        loss_sim, loss_nil = CountSmooth(max_steps=1000), CountSmooth(max_steps=1000)
        step = 0
        with D.guard(device):
            with tqdm(total=args.max_steps, desc='Train') as t:
                t.update(step)
                model.train()
                while True:
                    logger.info("EPOCH BEGIN...")
                    for model_inputs, raw_samples, labels_sim, labels_nil in train_loader.batch_reader():
                        # try:
                        step_info = forward_step(model_inputs, labels_sim, labels_nil)
                        step_info['step'] = step

                        # 打印log及平滑后的loss
                        writer.add_scalar('sim_train_loss', step_info['loss_sim'], step)
                        writer.add_scalar('nil_train_loss', step_info['loss_nil'], step)
                        loss_sim.add(step_info['loss_sim'])
                        loss_nil.add(step_info['loss_nil'])
                        step_info['loss_sim'] = loss_sim.get()
                        step_info['loss_nil'] = loss_nil.get()
                        step_info.pop('tp_sim')
                        step_info.pop('tp_nil')
                        t.set_postfix(step_info)

                        t.update(1)
                        step += 1

                        if step % args.save_steps == 0:
                            save_dir = join(args.output_dir, f'step_{step}')
                            logger.info(f"save to {save_dir}")
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)

                            eval_info = eval_model(args.eval_num)
                            eval_info['step'] = step
                            json.dump(eval_info, open(join(save_dir, "eval_info.json"), 'w'))
                            D.save_dygraph(model.state_dict(), join(save_dir, 'weights'))
                            # D.save_dygraph(optimizer.state_dict(), join(save_dir, 'weights'))
                            VERSION_CONFIG.dump(save_dir)
                            with open(join(save_dir, 'arguments.txt'), 'w', encoding='utf8') as f:
                                f.write(str(args))
                            if step >= args.max_steps:
                                return

    train()
    logger.info("正常退出！")


if __name__ == '__main__':
    main()
