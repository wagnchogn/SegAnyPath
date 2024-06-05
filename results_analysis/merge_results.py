import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
import time

pred_path = r'C:\Users\wagnchog\Desktop\path_sam\path_sam_results\sam_results\ablation_study_merge\bcss_sam_adapter'

exps = os.listdir(pred_path)

img_files = os.listdir(os.path.join(pred_path,exps[0],'sam-med2d','iter9_prompt'))
#true_path = '/root/autodl-tmp/dataset/camleyon17/split/split_mask'
true_path = r'C:\Users\wagnchog\Desktop\path_sam\path_sam_results\sam_results\bcss\ori\split_mask'
def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y
def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def SegMetrics(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric
metrics = ['iou', 'dice']

test_iter_metrics = [0] * len(metrics)

for img_file in tqdm(img_files):
    #mask_merge = []
    masks = []

    start_time = time.time()
    for exp in exps:
        img_path = os.path.join(pred_path,exp,'sam-med2d','iter9_prompt',img_file)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        masks.append(img)
    end_time = time.time()
    print(f"mask loading time: {end_time - start_time}")

    start_time = time.time()
    stacked_masks = np.stack(masks, axis=0)
    final_mask, _ = mode(stacked_masks, axis=0)
    final_mask = final_mask.squeeze(0)

    end_time = time.time()
    print(f"mask stacking time: {end_time - start_time}")

    start_time = time.time()
    label_file = cv2.imread(os.path.join(true_path,img_file),cv2.IMREAD_GRAYSCALE)
    ori_labels = label_file.reshape(1, 1, 256, 256)
    ori_labels = torch.from_numpy(ori_labels).to('cuda')

    final_mask = final_mask.reshape(1, 1, 256, 256)
    final_mask = torch.from_numpy(final_mask).to('cuda')
    test_batch_metrics = SegMetrics(final_mask, ori_labels, metrics)
    test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]
    end_time = time.time()

    print(f"mask evaluation time: {end_time - start_time}")
    for j in range(len(metrics)):
        test_iter_metrics[j] += test_batch_metrics[j]
l = len(img_files)
test_iter_metrics = [metric / l for metric in test_iter_metrics]
test_metrics = {metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
print(test_metrics)
    #print(final_mask)
        #print(img)
# 假设 masks 是一个包含所有分割掩码的列表或数组，形状为 [10, height, width]
# 每个掩码的形状为 [height, width]
#masks = [mask1, mask2, ..., mask10]  # 替换为您的掩码数据

# 将掩码堆叠成一个新的数组，形状为 [10, height, width]


# 应用多数投票法
#final_mask, _ = mode(stacked_masks, axis=0)
  # 移除多余的维度

# final_mask 现在是形状为 [height, width] 的数组，包含最终的分割结果
