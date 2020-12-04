import torch
import os

# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2

# original saved file with DataParallel
state_dict = torch.load('checkpoints/20201103-161225/resnet_face50_35.pth')

print(state_dict)

with open(os.path.join('./weight2_0.txt'), 'w') as f:
    f.write(f'{(state_dict)}\n')

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

torch.save(new_state_dict, 'checkpoints/20201103-161225/resnet_face50_35_m.pth')
print(new_state_dict)
with open(os.path.join('./weight2_1.txt'), 'w') as f:
    f.write(f'{(new_state_dict)}\n')