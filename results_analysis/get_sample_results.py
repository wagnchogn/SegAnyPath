import os
import json
import pandas as pd

data_route = '/root/autodl-tmp/code/SAM-Med2D/exps_bcss/results'

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

iou_results = []
dice_results = []

exps = os.listdir(data_route)
for exp in exps:
    exp_path = os.path.join(data_route, exp)
    result = load_json(os.path.join(exp_path, "9_1_point_metrics.json"))
    iou_results.append(result['iou'])
    dice_results.append(result['dice'])

df = pd.DataFrame({'exp':exps, 'iou':iou_results, 'dice':dice_results})
df.to_csv(os.path.join('/root/autodl-tmp/code/SAM-Med2D/exps_bcss/results', 'iou_dice.csv'))
