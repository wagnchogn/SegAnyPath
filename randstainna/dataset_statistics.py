import os
import cv2
import numpy as np
import time
import argparse
import yaml
import json
import random
import copy
from skimage import color
import pandas as pd
from fitter import Fitter #1.29添加，在统计的同时记录分布


parser = argparse.ArgumentParser(description='norm&jitter dataset lab statistics')
# Dataset
# 传入一个ImageFolder形式的数据集
parser.add_argument('--data_dir',type=str,metavar='DIR',default='/root/autodl-tmp/dataset/256_test_code/image2label_train.json',
                    help='path to dataset')
parser.add_argument('--dataset-name', type=str, default='/root/autodl-tmp/dataset/256_test_code',metavar='DIR',
                    help='dataset output_name')
parser.add_argument('--methods', type=str, default='Reinhard',
                    help='colornorm_methods')
parser.add_argument('--color-space', type=str, default=['LAB','HED','HSV'],choices=['LAB','HED','HSV'], #限定范围
                    help='dataset statistics color space')
parser.add_argument('--random', action='store_true', default=False,
                    help='random shuffle sample')
parser.add_argument('--n', type=int, default=0,metavar='DIR',
                    help='datasets statistics sample n image each class')

def _parse_args():

    args = parser.parse_args()

    return args

def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':

    args = _parse_args()

    path_dataset = args.data_dir



    t1 = time.time()
    i = 0

    json_data = load_json(path_dataset)
    img_path_all = list(json_data.keys())
    if args.random:
        random.shuffle(img_path_all)

    for sel_color in args.color_space:
        labL_avg_List = []
        labA_avg_List = []
        labB_avg_List = []
        labL_std_List = []
        labA_std_List = []
        labB_std_List = []
        # 原地打乱
        for img_path in img_path_all:
            if args.n == 0: # n=0表示统计全部的
                pass
            elif i < args.n:
                i += 1
            else:
                i = 0
                break
            #path_img = os.path.join(sub_path_class,image)

            img = cv2.imread(img_path)
            try:
                if sel_color == 'LAB':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                elif sel_color == 'HED': #12.20增加，统计HED信息
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #1.10发现bug，统计和最终的对应不上
                    img = color.rgb2hed(img)
                elif sel_color == 'HSV': #12.20增加，统计HSV信息
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                else:
                    print('wrong color space: {}!!'.format(sel_color))
                img_avg, img_std = getavgstd(img)
            except:
                print(img_path)
                continue

            labL_avg_List.append(img_avg[0])
            labA_avg_List.append(img_avg[1])
            labB_avg_List.append(img_avg[2])
            labL_std_List.append(img_std[0])
            labA_std_List.append(img_std[1])
            labB_std_List.append(img_std[2])
            #class_list.append(sub_class_dir)
        t2 = time.time()
        print(t2-t1)
        l_avg_mean = np.mean(labL_avg_List).item()
        l_avg_std = np.std(labL_avg_List).item()
        l_std_mean = np.mean(labL_std_List).item()
        l_std_std = np.std(labL_std_List).item()
        a_avg_mean = np.mean(labA_avg_List).item()
        a_avg_std = np.std(labA_avg_List).item()
        a_std_mean = np.mean(labA_std_List).item()
        a_std_std = np.std(labA_std_List).item()
        b_avg_mean = np.mean(labB_avg_List).item()
        b_avg_std = np.std(labB_avg_List).item()
        b_std_mean = np.mean(labB_std_List).item()
        b_std_std = np.std(labB_std_List).item()

        # 统计分布
        std_avg_list = [labL_avg_List, labL_std_List, labA_avg_List, labA_std_List, labB_avg_List, labB_std_List]
        distribution = []
        for std_avg in std_avg_list:
            f = Fitter(std_avg, distributions=['norm',  'laplace'])
            f.fit()
            distribution.append(list(f.get_best(method='sumsquare_error').keys())[0]) #分布关键词转成list之后取第一个即可

        yaml_dict_lab = {
            'random': args.random ,
            'n_each_class': args.n ,
            'color_space': sel_color,
            'methods': args.methods,
            '{}'.format(sel_color[0]):{   # lab-L/hed-H
                'avg':{
                    'mean': round(l_avg_mean,3) ,
                    'std': round(l_avg_std,3) ,
                    'distribution': distribution[0]
                },
                'std': {
                     'mean': round(l_std_mean,3),
                     'std': round(l_std_std,3),
                     'distribution': distribution[1]
                }
            },
            '{}'.format(sel_color[1]): {  # lab-A/hed-E
                'avg': {
                    'mean': round(a_avg_mean,3),
                    'std': round(a_avg_std,3),
                    'distribution': distribution[2]
                },
                'std': {
                    'mean': round(a_std_mean,3),
                    'std': round(a_std_std,3),
                    'distribution': distribution[3]
                }
            },
            '{}'.format(sel_color[2]): {  # lab-B/hed-D
                'avg': {
                    'mean': round(b_avg_mean,3),
                    'std': round(b_avg_std,3),
                    'distribution': distribution[4]
                },
                'std': {
                    'mean': round(b_std_mean,3),
                    'std': round(b_std_std,3),
                    'distribution': distribution[5]
                }
            }
        }
        #yaml_save_path = './{}.yaml'.format(args.dataset_name if args.dataset_name != '' else 'dataset_{}_random{}_n{}'.format(sel_color, args.random, args.n) )
        yaml_save_path = os.path.join(args.dataset_name, '{}.yaml'.format(sel_color))
        with open(yaml_save_path, 'w') as f:
            yaml.dump(yaml_dict_lab, f)
            print('The dataset lab statistics has been saved in {}'.format(yaml_save_path))