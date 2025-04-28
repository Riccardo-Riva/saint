import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset

# converts the time in seconds to hours, minutes and seconds (+ a text)
def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)

def data_split(X,Y,nan_mask=None):
    if nan_mask is not None:    
        x_d = {
            'data': np.array(X),
            'mask': np.array(nan_mask,dtype=np.int8)
        }
        
        if x_d['data'].shape != x_d['mask'].shape:
            raise'Shape of data not same as that of nan mask!'
        
    else:
        x_d = {
            'data': np.array(X),
            'mask': np.ones(X.shape,np.int8)
        }
            
    y_d = {
        'data': Y.reshape(-1, 1)
    } 
    return x_d, y_d

def data_split_indices(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        # subtracton between sets to exclude indices
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int32) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int32) #categorical columns mask
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int32) #numerical columns mask
        if task == 'clf':
            self.y = Y['data'] #.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int) # cls token
        self.cls_mask = np.ones_like(self.y,dtype=int) # cls token masking (it is always present)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

