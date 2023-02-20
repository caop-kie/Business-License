#!/usr/env/bin python3

"""
Generate training and test images.
"""
import os
import json

# prevent opencv use all cpus
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import traceback
import numpy as np

import multiprocessing as mp
from itertools import repeat

import cv2

from libs.config import load_config
from libs.timer import Timer
from parse_args import parse_args
import libs.utils as utils
import libs.font_utils as font_utils
from textrenderer.corpus.corpus_utils import corpus_factory
from textrenderer.renderer import Renderer
from tenacity import retry

lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'

flags = parse_args()
cfg = load_config(flags.config_file)

fonts = font_utils.get_font_paths_from_list(flags.fonts_list)
bgs = utils.load_bgs(flags.bg_dir)

corpus = corpus_factory(flags.corpus_mode, flags.chars_file, flags.corpus_dir, flags.length)

renderer = Renderer(corpus, fonts, bgs, cfg,
                    height=flags.img_height,
                    width=flags.img_width,
                    clip_max_chars=flags.clip_max_chars,
                    debug=flags.debug,
                    gpu=flags.gpu,
                    strict=flags.strict)


def start_listen(q, fname):
    """ listens for messages on the q, writes to file. """

    f = open(fname, mode='a', encoding='utf-8')
    while 1:
        m = q.get()
        if m == STOP_TOKEN:
            break
        try:
            f.write(str(m) + '\n')
        except:
            traceback.print_exc()

        with lock:
            if counter.value % 1000 == 0:
                f.flush()
    f.close()


# @retry
def gen_img_retry(renderer, img_index):
    try:
        return renderer.gen_img(img_index)
    except Exception as e:
        print("Retry gen_img: %s" % str(e))
        traceback.print_exc()
        raise Exception


def generate_img(img_index, q=None):
    global flags, lock, counter
    # Make sure different process has different random seed
    np.random.seed()

    company_name, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=None, bg=bgs[0], entity_type='name', entity_dict={}, ocr=[])  # 公司名称
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='location', entity_dict=entity_dict, ocr=ocr)  # 经营场所
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='operation_date', entity_dict=entity_dict, ocr=ocr)  # 成立日期，合伙日期
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='type', entity_dict=entity_dict, ocr=ocr)  # 公司类型
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='legal_person', entity_dict=entity_dict, ocr=ocr)  # 合伙人
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='business_scopes', entity_dict=entity_dict, ocr=ocr)  # 经营范围
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='social_credit_code', entity_dict=entity_dict, ocr=ocr)  # 信用代码
    _, im, word, entity_dict, ocr = renderer.gen_img_new(img_index, company_name=company_name, bg=im, entity_type='registration_capital', entity_dict=entity_dict, ocr=ocr)  # 注册资本
    ocr.append('142,350,356,350,356,378,142,378,统一社会信用代码,social_credit_code_key')
    ocr.append('142,533,288,533,288,565,142,565,名称,name_key')
    ocr.append('143,584,287,584,287,618,143,618,类型,type_key')
    ocr.append('850,691,1002,691,1002,723,850,723,住所,location_key')
    ocr.append('143,639,289,639,289,671,143,671,法定代表人,legal_person_key')
    ocr.append('845,531,1002,531,1002,564,845,564,注册资本,registration_capital_key')
    ocr.append('849,583,1001,583,1001,617,849,617,成立日期,start_date_key')
    ocr.append('849,636,1001,636,1001,671,849,671,营业期限,operation_date_key')
    ocr.append('144,690,291,690,291,722,144,722,经营范围,business_scopes_key')
    base_name = '{:03d}'.format(img_index)
    os.makedirs(os.path.join(flags.save_dir, 'boxes_and_transcripts'), exist_ok=True)
    os.makedirs(os.path.join(flags.save_dir, 'images'), exist_ok=True)
    with open(os.path.join(flags.save_dir, 'boxes_and_transcripts', base_name + '.tsv'), 'w', encoding='utf-8') as f:
        for i, line in enumerate(ocr):
            f.writelines(str(i) + ',' + line + '\n')
    if not flags.viz:
        fname = os.path.join(flags.save_dir, 'images', base_name + '.jpg')
        cv2.imwrite(fname, im)

        label = "{} {}".format(base_name, word)

        if q is not None:
            q.put(label)

        with lock:
            counter.value += 1
            print_end = '\n' if counter.value == flags.num_img else '\r'
            if counter.value % 100 == 0 or counter.value == flags.num_img:
                print("{}/{} {:2d}%".format(counter.value,
                                            flags.num_img,
                                            int(counter.value / flags.num_img * 100)),
                      end=print_end)
    else:
        utils.viz_img(im)


def sort_labels(tmp_label_fname, label_fname):
    lines = []
    with open(tmp_label_fname, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = sorted(lines)
    with open(label_fname, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line[9:])


def restore_exist_labels(label_path):
    # 如果目标目录存在 labels.txt 则向该目录中追加图片
    start_index = 0
    if os.path.exists(label_path):
        start_index = len(utils.load_chars(label_path))
        print('Generate more text images in %s. Start index %d' % (flags.save_dir, start_index))
    else:
        print('Generate text images in %s' % flags.save_dir)
    return start_index


def get_num_processes(flags):
    processes = flags.num_processes
    if processes is None:
        processes = max(os.cpu_count(), 2)
    return processes


if __name__ == "__main__":
    # It seems there are some problems when using opencv in multiprocessing fork way
    # https://github.com/opencv/opencv/issues/5150#issuecomment-161371095
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    if utils.get_platform() == "OS X":
        mp.set_start_method('spawn', force=True)

    if flags.viz == 1:
        flags.num_processes = 1

    tmp_label_path = os.path.join(flags.save_dir, 'tmp_labels.txt')
    label_path = os.path.join(flags.save_dir, 'labels.txt')
    # Synthesize business licenses with index from 329 to 499.
    last_index = 328
    for i in range(329, 500):
        try:
            generate_img(last_index + 1)
        except Exception as e:
            continue
        last_index = last_index + 1
