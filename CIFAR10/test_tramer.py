import sys
sys.path.append('./models/')
import torch
from models import PreActResNet18
import numpy as np
from wideresnet import WideResNet

import torch.optim as optim
import torch.nn as nn
sys.path.append('./utils/')
from core import *
from torch_backend import *
import ipdb
import sys 
import foolbox as fb
import foolbox.attacks as fa
from cifar_funcs import *
import argparse
from time import time
# from fast_adv.attacks import DDN

# python3 test.py -gpu_id 0 -model 0 -batch_size 1 -attack 0 -restarts 10

parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-batch_size", help = "Batch Size for Test Set (Default = 1000)", type = int, default = 100)
parser.add_argument("-attack", help = "Foolbox = 0; Custom PGD = 1", type = int, default = 0)
parser.add_argument("-restarts", help = "Default = 1", type = int, default = 1)
parser.add_argument("-path", help = "To override default model fetching- Automatically appends '.pt' to path", type = str)
parser.add_argument("-subset", help = "Subset of attacks", type = int, default = -1)
parser.add_argument("-epsilon_l_1", help = "Eps", type = float, default = 2000/255.)
parser.add_argument("-epsilon_l_2", help = "Eps", type = float, default = 0)
parser.add_argument("-epsilon_l_inf", help = "Eps", type = float, default = 4/255.)
parser.add_argument("-alpha_l_1", help = "Eps", type = float, default = 0.5)
parser.add_argument("-alpha_l_2", help = "Eps", type = float, default = 0)
parser.add_argument("-alpha_l_inf", help = "Eps", type = float, default = 0.001)


params = parser.parse_args()

device_id = params.gpu_id
batch_size = params.batch_size
attack = params.attack
res = params.restarts
path = params.path
subset = params.subset


device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))


DATA_DIR = './data'
dataset = cifar10(DATA_DIR)
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
test_loader = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())

version = int(fb.__version__.split(".")[0])
print(version)

def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'PA':
        metric = 0
        # A = fa.L0BrendelBethgeAttack()
        A = fa.PointwiseAttack()
        if version <=2:
            A = fa.PointwiseAttack(fmodel, distance = fb.distances.L0)
        else:
            raise("Incompatible Version of Foolbox")
    elif attack == "EAD":
        A = fa.EADAttack()
    elif 'PGD' in attack:
        A = fa.LinfPGD()
    return A, 0,0,0


def parse_subset(subset):
    if subset == 0:
        attacks_list = ['PA']
        types_list   = [ 1 ]
    elif subset == 1 :
        attacks_list =['PGD','EAD']
        types_list = [3,1]
    elif subset == 2 :
        attacks_list =['EAD']
        types_list = [1]
    return attacks_list, types_list    

def test_foolbox(model_name, max_tests, model_test):
    #Saves the minimum epsilon value for successfully attacking each image via different foolbox attacks as an npy file in the folder corresponding to model_name
    #No Restarts in case of BA
    print(max_tests, model_name)
    torch.manual_seed(0)
    model_test.eval()
    preprocessing = dict()
    bounds = (0, 1)
    # ipdb.set_trace()
    if version  <= 2:
        fmodel = fb.models.PyTorchModel(model_test,bounds=(0., 1.), num_classes=10, device=device)
    else:
        fmodel = fb.PyTorchModel(model_test, bounds=bounds, preprocessing=preprocessing, device = device)
        fmodel = fmodel.transform_bounds((0, 1))
        assert fmodel.bounds == (0, 1)

    attacks_list, types_list = parse_subset(subset)
    norm_dict = {0:norms_l1_squeezed, 1:norms_l1_squeezed, 2:norms_l2_squeezed,3:norms_linf}
    norm_full_dict = {0:norms_l1, 1:norms_l1, 2:norms_l2,3:norms_linf}
    epsilons_dict = {0:2000/255., 1:2000/255., 2:0, 3:4/255.}
    attacks_dist = []

    for i in range(len(attacks_list)):
        test_loader = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())
        # ipdb.set_trace()
        attack_name = attacks_list[i]
        restarts = res
        print (attack_name)
        types = types_list[i]
        norm = norm_dict[types]
        norm_full = norm_full_dict[types]

        start = time()
        output = np.ones((max_tests))
        
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        for batches in test_loader:        
            images = batches['input'].to(device)
            labels = batches['target'].to(device)
            start = time()
            
            distance = 1000*torch.ones(batch_size).to(device)
            advs_final = images.clone()
            for r in range (restarts):
                if version <= 2:
                    advs = attack(images.cpu().numpy(), labels=labels.cpu().numpy())
                    advs = torch.from_numpy(advs).to(device)
                else:
                    criterion = fb.criteria.Misclassification(labels)
                    advs, clipped, is_adv = attack(fmodel, images, criterion, epsilons=epsilons_dict[types])
                new_distance = norm(images-advs)
                advs_final[distance > new_distance] = advs[distance>new_distance]
                distance[distance > new_distance] = new_distance[distance>new_distance]

                
            if version <=2 :
                deltas = advs_final - images
                deltas_final = epsilons_dict[types]* deltas/norm_full(deltas).clamp(min=epsilons_dict[types])
                cut_distance = norm(deltas_final)
                # print(distance)
                advs_final = images + deltas_final
            classes = model_test(advs_final)
            num_correct = classes.max(1)[1] == labels
            output[total:total+batch_size] = distance.cpu()
            total += batch_size
            print(total, " ", attack_name, " " ,model_name, " distance = ", distance.mean(), " num correct = ", num_correct.sum().item(), " Time taken = ", time() - start,)
        

            if (total >= max_tests):
                np.save(model_name + "/" + attack_name + ".npy" ,output)
                attacks_dist.append(output)
                break

        print("Time Taken = ", time() - start)
    return attacks_dist



def test_pgd(model, model_name, clean = False):
    print (model_name)
    print(device)
    model.eval()
    total_loss, total_acc_inf = epoch_adversarial(test_loader, None,  model, None, pgd_linf, 
                             device = device, stop = True, restarts = 0, randomize = 1,
                             epsilon = 4/255., alpha = 0.08/255., num_iter =100)
    print('Test Acc Inf: {0:.4f}'.format(total_acc_inf))    

    total_loss, total_acc_1 = epoch_adversarial(test_loader, None,  model, None, pgd_l1_topk, 
                            device = device, stop = True, restarts = 1, randomize = 0,
                            epsilon = 7.843, alpha = 0.5, num_iter = 50)
                            # epsilon = 2000/255., alpha = 200/255., num_iter = 50)
    print('Test Acc 1: {0:.4f}'.format(total_acc_1))    

assert (path is not None)
model_name = path

import os
if(not os.path.exists(model_name)):
    os.makedirs(model_name)

# model = PreActResNet18().to(device)
# for m in model.children(): 
#     if not isinstance(m, nn.BatchNorm2d):
#         m.float()   
model = WideResNet().to(device)

criterion = nn.CrossEntropyLoss()

start_time = time()

model.load_state_dict(torch.load(model_name+".pt", map_location = device))
model.eval()

print (device)
if attack == 0:
    attacks_dist_list = test_foolbox(model_name, 1000, model)
    ipdb.set_trace()
elif attack == 1:
    test_pgd(model,model_name)
