import torch
import os

# 참고] https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
# Data.Paralle을 사용시에 weight가 module에 싸여서 저장된다. 

# original saved file with DataParallel
state_dict = torch.load('checkpoints/20201102-151532/resnet_face50_9.pth')

print(state_dict)

with open(os.path.join('./weight2_0.txt'), 'w') as f:
    f.write(f'{(state_dict)}\n')

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

print(new_state_dict)
with open(os.path.join('./weight2_1.txt'), 'w') as f:
    f.write(f'{(new_state_dict)}\n')
    
 
