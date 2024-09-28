import pandas as pd
from torch.utils.data import DataLoader
from Dataset import FN_Dataset
from model import FN_Generator
from Dataset import sp
import torch
import torch.nn as nn
import torch.optim as optim

EPOCH = 100
batchsize = 32


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
                        batch_size=batchsize,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)

generator = FN_Generator(output_dim=sp.vocab_size(),
                         headline_vocabsize=sp.vocab_size(),
                         inputsize_H=dataset.maxlen_H,
                         ).to(device)

crossentropy = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(generator.parameters(), lr=0.005)

for epoch in range(EPOCH):
    for i, (src_h, src_o, tgt) in enumerate(dataloader):
        for l in range(1, dataset.maxlen_S):
          optimizer.zero_grad(set_to_none=True)
          output = generator(src_h.to(device),
                            src_o.to(device),
                            tgt[:, 0:l].to(device))

          loss = crossentropy(output,
                              nn.functional.one_hot(tgt[:, l].to(device),
                                                            num_classes=sp.vocab_size()).to(torch.float32))
          loss.backward()
          optimizer.step()
    print('Epoch',epoch,' Loss -->',loss.item())