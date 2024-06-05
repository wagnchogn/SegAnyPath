
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random
from skimage import color
from typing import Optional, Dict
import yaml

class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        dataset = json.load(json_file)
        mag_infos = json.load(open(os.path.join(data_path, f'label2mag_{mode}.json'), "r"))
        task_infos = json.load(open(os.path.join(data_path, f'{mode}_task_dict.json'), "r"))

        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
        self.mags = list(mag_infos.values())
        self.task_infos = list(task_infos.values())

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        #print(self.label_paths[index])
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])
        mag_label = mags_dict.get(self.mags[index])
        task_label = self.task_infos[index]

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask,img_path = self.label_paths[index])
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        image_input["mag_label"] = torch.tensor(mag_label)
        image_input['task_label'] = torch.tensor(task_label)

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)

def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    # str->dict
    data = yaml.load(file_data, Loader=yaml.FullLoader)

    return data


class Dict2Class(object):

    def __init__(
            self,
            my_dict: Dict
    ):
        self.my_dict = my_dict
        for key in my_dict:
            setattr(self, key, my_dict[key])

class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        mag_infos = json.load(open(os.path.join(data_dir, f'image2mag_{mode}.json'), "r"))
        task_infos = json.load(open(os.path.join(data_dir, f'{mode}_task_dict.json'), "r"))

        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
        self.mags = list(mag_infos.values())
        self.tasks = list(task_infos.values())

        self.randstainna_data = get_yaml_data(os.path.join(data_dir, 'Random(HED+LAB+HSV)_n0.yaml'))
        self.p = 1.0
        self.std_adjust = 0
        #self.color_space = c_s
        self.distribution = 'normal'

    def _getavgstd(
            self,
            image: np.ndarray,
            isReturnNumpy: Optional[bool] = True
    ):

        avgs = []
        stds = []

        num_of_channel = image.shape[2]
        for idx in range(num_of_channel):
            avgs.append(np.mean(image[:, :, idx]))
            stds.append(np.std(image[:, :, idx]))

        if isReturnNumpy:
            return (np.array(avgs), np.array(stds))
        else:
            return (avgs, stds)

    def _normalize(
            self,
            img: np.ndarray,
            img_avgs: np.ndarray,
            img_stds: np.ndarray,
            tar_avgs: np.ndarray,
            tar_stds: np.ndarray,
            color_space: str
    ) -> np.ndarray:

        img_stds = np.clip(img_stds, 0.0001, 255)
        img = (img - img_avgs) * (tar_stds / img_stds) + tar_avgs

        if color_space in ["LAB", "hSV"]:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def augment(self,image,color_space):
        num_of_channel = image.shape[2]
        channel_avgs = {
            'avg': [self.randstainna_data[color_space[0]]['avg']['mean'],
                    self.randstainna_data[color_space[1]]['avg']['mean'],
                    self.randstainna_data[color_space[2]]['avg']['mean']],
            'std': [self.randstainna_data[color_space[0]]['avg']['std'],
                    self.randstainna_data[color_space[1]]['avg']['std'],
                    self.randstainna_data[color_space[2]]['avg']['std']]
        }
        channel_stds = {
            'avg': [self.randstainna_data[color_space[0]]['std']['mean'],
                    self.randstainna_data[color_space[1]]['std']['mean'],
                    self.randstainna_data[color_space[2]]['std']['mean']],
            'std': [self.randstainna_data[color_space[0]]['std']['std'],
                    self.randstainna_data[color_space[1]]['std']['std'],
                    self.randstainna_data[color_space[2]]['std']['std']],
        }
        channel_avgs = Dict2Class(channel_avgs)
        channel_stds = Dict2Class(channel_stds)
        if color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif color_space == 'hSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'HED':
            image = color.rgb2hed(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

        std_adjust = self.std_adjust

        # virtual template generation
        tar_avgs = []
        tar_stds = []
        if self.distribution == 'uniform':
            # three-sigma rule for uniform distribution
            for idx in range(num_of_channel):
                tar_avg = np.random.uniform(
                    low=channel_avgs.avg[idx] - 3 * channel_stds.std[idx],
                    high=channel_avgs.avg[idx] - 3 * channel_stds.std[idx]
                )
                tar_std = np.random.uniform(
                    low=channel_avgs.avg[idx] - 3 * channel_stds.std[idx],
                    high=channel_avgs.avg[idx] - 3 * channel_stds.std[idx]
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
        else:
            if self.distribution == 'normal':
                np_distribution = np.random.normal
            elif self.distribution == 'laplace':
                np_distribution = np.random.laplace

            for idx in range(num_of_channel):
                tar_avg = np_distribution(
                    loc=channel_avgs.avg[idx],
                    scale=channel_avgs.std[idx] * (1 + std_adjust)
                )

                tar_std = np_distribution(
                    loc=channel_stds.avg[idx],
                    scale=channel_stds.std[idx] * (1 + std_adjust)
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)

        tar_avgs = np.array(tar_avgs)
        tar_stds = np.array(tar_stds)

        img_avgs, img_stds = self._getavgstd(image)

        image = self._normalize(
            img=image,
            img_avgs=img_avgs,
            img_stds=img_stds,
            tar_avgs=tar_avgs,
            tar_stds=tar_stds,
            color_space=color_space
        )

        if color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif color_space == 'hSV':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif color_space == 'HED':
            nimg = color.hed2rgb(image)
            imin = nimg.min()
            imax = nimg.max()
            rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]

            image = cv2.cvtColor(rsimg, cv2.COLOR_RGB2BGR)

        return image
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            #image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        color_space = random.choice(['LAB', 'hSV', 'none'])
        if color_space!= 'none':
            image = self.augment(image,color_space)
        mag_label = mags_dict.get(self.mags[index])
        task_label = self.tasks[index]
        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
        image = (image - self.pixel_mean) / self.pixel_std
        masks_list = []

        boxes_list = []
        point_coords_list, point_labels_list = [], []

        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)


        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)
        #mag_labels = torch.tensor(mags_list)
        mag_labels = torch.tensor(mag_label)
        task_labels = torch.tensor(task_label)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input['mag_label'] = mag_labels
        image_input['task_label'] = task_labels
        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

mags_dict = {'5x':0,'10x':1,'20x':2,'40x':3,'None':4}

if __name__ == "__main__":
    train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

