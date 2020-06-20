import sys
sys.path.append('./models/')
import torch
from models import PreActResNet18
from wideresnet import WideResNet
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
sys.path.append('./utils/')
from core import *
from torch_backend import *
from cifar_funcs import *
import ipdb
import sys 
import argparse

# python3 train.py -gpu_id 0 -model 3 -batch_size 128 -lr_schedule 1
parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: avg \n\t 5: max \n\t 6: vanilla", type=int, default = 3)
parser.add_argument("-batch_size", help = "Batch Size for Train Set", type = int, default = 100)
parser.add_argument("-epsilon_l_1", help = "Eps", type = float, default = 12)
parser.add_argument("-epsilon_l_2", help = "Eps", type = float, default = 0.5)
parser.add_argument("-epsilon_l_inf", help = "Eps", type = float, default = 0.031)
parser.add_argument("-alpha_l_1", help = "Step Size", type = float, default = 1.0)
parser.add_argument("-alpha_l_2", help = "Step Size", type = float, default = 0.05)
parser.add_argument("-alpha_l_inf", help = "Step Size", type = float, default = 0.003)
parser.add_argument("-lr_max", help = "lr_max", type = float, default = 0.1)
parser.add_argument("-num_iter", help = "Iterations", type = int, default = 50)
parser.add_argument("-model_id", help = "Id", type = int, default = 0)
parser.add_argument("-resume", help = "resume", type = int, default = 0)
parser.add_argument("-resume_iter", help = "resume", type = int, default = -1)
parser.add_argument("-epochs", help = "resume", type = int, default = 50)
parser.add_argument("-model_type", help = "architecture", type = str, default = "preactresnet")

params = parser.parse_args()
device_id = params.gpu_id

device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

torch.cuda.device_count() 
batch_size = params.batch_size
model_type = params.model_type
lr_max = params.lr_max
choice = params.model


epochs = params.epochs
DATA_DIR = './data'
dataset = cifar10(DATA_DIR)

train_set = list(zip(transpose(normalise2(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])
train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2, gpu_id = torch.cuda.current_device())
test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())


if model_type == "preactresnet":
    model = PreActResNet18().to(device)
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half()   
else:
    #wrn-28-10
    splits = model_type.split("-")
    depth = int(splits[1])
    widen_factor = int(splits[2])
    model = WideResNet( depth = depth, 
                        widen_factor = widen_factor).to(device) 


opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

import time

lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, lr_max, lr_max/10.0, 0])[0]

attack_list = [ pgd_linf ,  pgd_l1_topk,   pgd_l2 ,  msd_v0 ,  triple_adv ,  pgd_worst_dir, triple_adv]#TRIPLE, VANILLA DON'T HAVE A ATTACK NAME ANYTHING WORKS
attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "msd_v0", "triple_adv", "pgd_worst_dir", "vanilla"]
folder_name = ["LINF", "L1", "L2", "MSD", "AVG", "MAX", "VANILLA"]

model_dir = f"Models/{folder_name[choice]}/{params.model_id}"
import os
if(not os.path.exists(model_dir)):
    os.makedirs(model_dir)

file = open("{0}/logs.txt".format(model_dir), "w")

with open(f"{model_dir}/model_info.txt", "w") as f:
    import json
    json.dump(params.__dict__, f, indent=2)

def myprint(a):
    print(a)
    file.write(a)
    file.write("\n")

attack = attack_list[choice]
print(attack_name[choice])

t_start = 1
if params.resume:
    resume_iter = params.resume_iter
    location = f"{model_dir}/iter_{resume_iter}.pt"
    print(location)
    t_start = resume_iter + 1
    model.load_state_dict(torch.load(location, map_location = device))

for epoch_i in range(t_start,epochs+1):  
    print(epoch_i)
    start_time = time.time()
    lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_batches))
    if choice == 6:
        train_loss, train_acc = epoch(train_batches, lr_schedule, model, epoch_i, criterion, opt = opt, device = device)
    elif choice == 4:
        train_loss, train_acc = triple_adv(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.3)
    elif choice == 3:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, 
            epsilon_l_1 = params.epsilon_l_1, epsilon_l_2 = params.epsilon_l_2, epsilon_l_inf = params.epsilon_l_inf,
            alpha_l_1 = params.alpha_l_1, alpha_l_2 = params.alpha_l_2, alpha_l_inf = params.alpha_l_inf,
            num_iter = params.num_iter)

    elif choice == 5:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.3, alpha_l_inf = 0.005)
    else:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device)

    total_loss, total_acc   = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = "cuda:1")
    total_loss, total_acc_1 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l1_topk, criterion, opt = None, device = device, stop = True, 
                                    epsilon = params.epsilon_l_1)
    total_loss, total_acc_2 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True, 
                                    epsilon = params.epsilon_l_2) if params.epsilon_l_2 != 0 else 0,100
    total_loss, total_acc_3 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True, 
                                    epsilon = params.epsilon_l_inf)
    myprint('Epoch: {7}, Clean Acc: {6:.4f} Train Acc: {5:.4f}, Test Acc 1: {4:.4f}, Test Acc 2: {3:.4f}, Test Acc inf: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3, total_acc_2,total_acc_1,train_acc, total_acc, epoch_i))    
    if epoch_i %5 == 0:
        torch.save(model.state_dict(), "{0}/iter_{1}.pt".format(model_dir, str(epoch_i)))
