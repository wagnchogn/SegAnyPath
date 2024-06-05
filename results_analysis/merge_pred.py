import os
import cv2
import numpy as np

pred_route = r'D:\code\sam_results\exps_1212\results_show\modes1'
save_route = r'D:\code\sam_results\exps_1212\results_show\merge'
if not os.path.exists(save_route):
    os.makedirs(save_route)

prompts = ['iter8_prompt']
for prompt in prompts:
    save_path = os.path.join(save_route,prompt)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    modes = os.listdir(pred_route)

    img_names = os.listdir(os.path.join(pred_route,modes[1],prompt))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    text_padding = 10  # 文本与图像顶部的间隔

    for img_name in img_names:
        images = []  # 存储图像的列表
        texts = []  # 存储文件夹名称的列表
        images_with_text = []
        for mode in modes:
            texts.append(mode)
            if mode == '0_label':
                img_path = os.path.join(pred_route, mode, img_name)
            else:
                img_path = os.path.join(pred_route, mode, prompt,img_name)
            img = cv2.imread(img_path)
            if img is not None:
                text = mode
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # 创建足够空间的画布以包含文本和图像
                canvas = np.zeros((img.shape[0] + text_height + text_padding, img.shape[1], img.shape[2]),
                                  dtype=np.uint8)
                canvas[text_height + text_padding:, :] = img

                # 在画布上添加文本
                text_org = (10, text_height + text_padding // 2)
                cv2.putText(canvas, text, text_org, font, font_scale, font_color, font_thickness)

                # 存储添加了文本的图像
                images_with_text.append(canvas)
            else:
                print(f"Image not found in the folder {mode}")

        combined_image = cv2.hconcat (images_with_text)

        cv2.imwrite(os.path.join(save_path,img_name), combined_image)

