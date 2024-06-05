import os
import json
import pandas as pd

data_route = '/root/autodl-tmp/code/SAM-Med2D'

save_route = os.path.join(data_route, 'exps_camelyon17')

pred_route = os.path.join(data_route, 'exps_camelyon17')

modes = os.listdir(pred_route)

if not os.path.exists(save_route):
    os.makedirs(save_route)
def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

boxes_dice_results = []
boxes_iou_results = []
one_point_dice_results = []
one_point_iou_results = []
three_point_dice_results = []
three_point_iou_results = []
five_point_dice_results = []
five_point_iou_results = []
nine_point_dice_results = []
nine_point_iou_results = []
one_eight_point_dice_results = []
one_eight_point_iou_results = []



for mode in modes:
    mode_path = os.path.join(pred_route, mode)

    boxes_result = load_json(os.path.join(mode_path, '_boxes_metrics.json'))

    one_point_result = load_json(os.path.join(mode_path,'1_1_point_metrics.json'))
    three_point_result = load_json(os.path.join(mode_path,'3_1_point_metrics.json'))
    five_point_result = load_json(os.path.join(mode_path,'5_1_point_metrics.json'))
    nine_point_result = load_json(os.path.join(mode_path,'9_1_point_metrics.json'))

    one_eight_point_result = load_json(os.path.join(mode_path,'1_8_point_metrics.json'))

    boxes_dice_results.append(boxes_result['dice'])
    boxes_iou_results.append(boxes_result['iou'])
    one_point_dice_results.append(one_point_result['dice'])
    one_point_iou_results.append(one_point_result['iou'])
    three_point_dice_results.append(three_point_result['dice'])
    three_point_iou_results.append(three_point_result['iou'])
    five_point_dice_results.append(five_point_result['dice'])
    five_point_iou_results.append(five_point_result['iou'])
    nine_point_dice_results.append(nine_point_result['dice'])
    nine_point_iou_results.append(nine_point_result['iou'])
    one_eight_point_dice_results.append(one_eight_point_result['dice'])
    one_eight_point_iou_results.append(one_eight_point_result['iou'])

boxes_df = pd.DataFrame({'mode': modes,'dice': boxes_dice_results, 'iou': boxes_iou_results})
one_point_df = pd.DataFrame({'mode': modes,'dice': one_point_dice_results, 'iou': one_point_iou_results})
three_point_df = pd.DataFrame({'mode': modes,'dice': three_point_dice_results, 'iou': three_point_iou_results})
five_point_df = pd.DataFrame({'mode': modes,'dice': five_point_dice_results,'iou': five_point_iou_results})
nine_point_df = pd.DataFrame({'mode': modes,'dice': nine_point_dice_results,'iou': nine_point_iou_results})
one_eight_point_df = pd.DataFrame({'mode': modes,'dice': one_eight_point_dice_results,'iou': one_eight_point_iou_results})

boxes_df.to_csv(os.path.join(save_route, 'boxes_metrics.csv'), index=False)
one_point_df.to_csv(os.path.join(save_route, '1_1_point_metrics.csv'), index=False)
three_point_df.to_csv(os.path.join(save_route, '3_1_point_metrics.csv'), index=False)
five_point_df.to_csv(os.path.join(save_route, '5_1_point_metrics.csv'), index=False)
nine_point_df.to_csv(os.path.join(save_route, '9_1_point_metrics.csv'), index=False)
one_eight_point_df.to_csv(os.path.join(save_route, '1_8_point_metrics.csv'), index=False)

with pd.ExcelWriter(os.path.join(save_route, 'all.xlsx')) as writer:
    boxes_df.to_excel(writer, sheet_name='boxes_metrics')
    one_point_df.to_excel(writer, sheet_name='1_1_point_metrics')
    three_point_df.to_excel(writer, sheet_name='3_1_point_metrics')
    five_point_df.to_excel(writer, sheet_name='5_1_point_metrics')
    nine_point_df.to_excel(writer, sheet_name='9_1_point_metrics')
    one_eight_point_df.to_excel(writer, sheet_name='1_8_point_metrics')