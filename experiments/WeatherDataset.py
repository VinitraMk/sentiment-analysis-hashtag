import torch
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
from constants.model_enums import Model
import re

#custom imports
from helper.utils import get_model_params, get_target_cols

class WeatherDataset(Dataset):

    def __init__(self,input_path, is_train = True):
        self.root_dir = os.environ["ROOT_DIR"]
        self.data = pd.read_csv(input_path)
        self.target_cols = get_target_cols()
        if is_train:
            self.X = self.data.drop(columns=self.target_cols)
            self.y = self.data[self.target_cols]

    def __prepare_hashtags(self, text):
        return re.findall(r"#(\w+)", text)
    
    def __prepare_label(self, label_list, sample):
        for (idx, x) in enumerate(self.target_cols):
            sample[x] = label_list[idx]
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_args = get_model_params()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is not None:
            label = (self.y.iloc[idx]).to_numpy()
            hashtags = self.__prepare_hashtags(self.X.iloc[idx]['tweet'])
        else:
            label = [0] * 24

        #label = self.y.iloc[idx] if self.y is not None else -1
        text = self.X.iloc[idx]['tweet']
        id = self.X.iloc[idx]['id']
        sample = { "tweet": text, "id": id, "hashtags": hashtags }
        sample = self.__prepare_label(label, sample)
        return sample

