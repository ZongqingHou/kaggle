#!/usr/bin/env python
# coding: utf-8

# ## self defined dataset

# In[2]:


from PIL import Image
import pandas as pd
from torch.utils import data


# In[1]:


class Blindness(data.Dataset):
    def __init__(self, root_path, data_path, csv_flag, tranform=None, target_tranform=None):
        if csv_flag is True:
            self.dataset_information = pd.read_csv(data_path)
        else:
            pass
        
        self.root_path = root_path
        self.tranform = tranform
        self.target_tranform = target_tranform
    
    def __getitem__(self, index):
        tmp_img_name, tmp_label = self.dataset_information.loc[index]
        
        img_path = self.root_path + tmp_img_name + ".png" if self.root_path[-1] == '/' else self.root_path + '/' + tmp_img_name + ".png"
        img_data = Image.open(img_path).convert("RGB")
        
        if self.tranform is not None:
            img_data = self.tranform(img_data)
        
        if self.target_tranform is not None:
            tmp_label = self.target_tranform(tmp_label)
            
        return img_data, tmp_label
    
    def __len__(self):
        return self.dataset_information.shape[0]

