import foolbox as fb
import foolbox.attacks as fa
import numpy as np
import torch
import ipdb
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mnist_funcs import *
from time import time
import argparse 
# from fast_adv.attacks import DDN

version = int(fb.__version__.split(".")[0])

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

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
        A = fa.BoundaryAttack()
    return A, 0,0,0


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

    attacks_list = ['PA']
    types_list   = [ 0]
    norm_dict = {0:norms_l1_squeezed, 1:norms_l1_squeezed, 2:norms_l2_squeezed,3:norms_linf}
    norm_full_dict = {0:norms_l1, 1:norms_l1, 2:norms_l2,3:norms_linf}
    epsilons_dict = {0:10, 1:10, 2:2.0, 3:0.03}
    attacks_dist = []

    for i in range(len(attacks_list)):
        test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
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
        for images,labels in test_loader:        
            #ipdb.set_trace()
            images = images.to(device)
            labels = labels.to(device)
            start = time()
            # ipdb.set_trace()
            
            distance = 1000*torch.ones(batch_size).to(device)
            advs_final = images.clone()
            for r in range (restarts):
                # try:
                if version <= 2:
                    advs = attack(images.cpu().numpy(), labels=labels.cpu().numpy())
                    advs = torch.from_numpy(advs).to(device)
                else:
                    criterion = fb.criteria.Misclassification(labels)
                    advs, clipped, is_adv = attack(fmodel, images, criterion, epsilons=epsilons_dict[types])
                new_distance = norm(images-advs)
                advs_final[distance > new_distance] = advs[distance>new_distance]
                distance[distance > new_distance] = new_distance[distance>new_distance]
                # except:
                #     a = 1
                
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


def test_pgd_saver(model_name, model):
    #Saves the minimum epsilon value for successfully attacking each image via PGD based attack as an npy file 
    #in the folder corresponding to model_name
    eps_1 = [10] #[3,6,(10),12,20,30,50,60,70,80,90,100]
    eps_2 = [2]  #[0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3,5,7,10]
    eps_3 = [0.3]#[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    num_1 = [100]#[50,50,100,100,100,200,200,200,300,300,300,300]
    num_2 = [100]#[30,40,50,50,100,100,150,150,150,150,300,300]
    num_3 = [100]#[30,40,50,50,100,100,150,150,150,150,300,300]
    attacks_l1 = torch.ones((batch_size, 12))*1000
    attacks_l2 = torch.ones((batch_size, 12))*1000
    attacks_linf = torch.ones((batch_size, 12))*1000    
    res = 1    

    for index in range(len(eps_1)):
        test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
        e_1 = eps_1[index]
        n_1 = num_1[index]
        eps, total_acc_1 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l1_topk, e_1, n_1, device = device, restarts = res)
        # eps, total_acc_1 = epoch_adversarial(test_batches, 0, model, 0, msd_v0, stop = True)
        # print(1/0)
        attacks_l1[:,index] = eps
        
    attacks_l1 = torch.min(attacks_l1,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL1" + ".npy" ,attacks_l1.numpy())

    for index in range(len(eps_2)):        
        test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
        e_2 = eps_2[index]
        n_2 = num_2[index]
        eps, total_acc_2 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l2, e_2, n_2, device = device, restarts = res)
        attacks_l2[:,index] = eps
    attacks_l2 = torch.min(attacks_l2,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL2" + ".npy" ,attacks_l2.numpy())

    for index in range(len(eps_3)):
        test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
        e_3 = eps_3[index]
        n_3 = num_3[index]
        eps, total_acc_3 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_linf, e_3, n_3, device = device, restarts = res)
        attacks_linf[:,index] = eps
    attacks_linf = torch.min(attacks_linf,dim = 1)[0]
    np.save(model_name + "/" + "CPGDLINF" + ".npy" ,attacks_linf.numpy())

    return attacks_l1, attacks_l2, attacks_linf



# python3 evaluate_grid_search.py -gpu_id 0 -path 

parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-batch_size", help="Id of GPU to be used", type=int, default = 1000)
parser.add_argument("-path", help = "To override default model fetching- Automatically appends '.pt' to path", type = str)
parser.add_argument("-res", help = "restarts", type = int, default = 1)
params = parser.parse_args()

device_id = params.gpu_id
path = params.path
batch_size = params.batch_size
res = params.res
mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

model_name = path
print(path)
import os
if(not os.path.exists(model_name)):
    os.makedirs(model_name)

model = net().to(device)
model_address = model_name + ".pt"
# try:
model.load_state_dict(torch.load(model_address, map_location = device))
# except:
    # model_address = "Models/MSD/model_1/iter_7.pt"
    # model.load_state_dict(torch.load(model_address, map_location = device))
model.eval()

def myprint(a):
    a = str(a)
    print(a)
    file.write(a)
    file.write("\n")

file = open("{0}/pgd_test.txt".format(model_name), "a")
# ipdb.set_tracce()
attacks_PA, = test_foolbox(model_name, 1000, model)
myprint(attacks_PA[attacks_PA>10].shape[0]/10)
# myprint(attacks_BA[attacks_BA>2.0].shape[0]/10)
attacks_l1, attacks_l2, attacks_linf = test_pgd_saver(model_name, model)

myprint(attacks_l1[attacks_l1>10].shape[0]/10)
myprint(attacks_l2[attacks_l2>2.0].shape[0]/10)
myprint(attacks_linf[attacks_linf>0.3].shape[0]/10)

a_all = np.vstack((attacks_PA>10, attacks_l1>10, attacks_l2>2.0, attacks_linf>0.3)).min(axis =0)    
myprint(a_all.sum()/10)

