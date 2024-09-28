import pandas as pd
from torch.utils.data import DataLoader
from dataset import FN_Dataset
from dataset import sp
import torch

directory = "./News_Category_Dataset_v3.json"
data = pd.read_json(directory, lines=True)
data = data.drop('link', axis=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nCurrent device using:", device)

dataset = FN_Dataset(df=data,
                     bos_token=sp.bos_id(),
                     eos_token=sp.eos_id(),
                     pad_token=sp.unk_id(),
                     )
dataloader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)