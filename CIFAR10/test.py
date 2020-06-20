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
# import foolbox as fb
# import foolbox.attacks as fa
from cifar_funcs import *
import argparse
from time import time
# from fast_adv.attacks import DDN

# python3 test.py -gpu_id 0 -model 0 -batch_size 1 -attack 0 -restarts 10

parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int, default = 3)
parser.add_argument("-batch_size", help = "Batch Size for Test Set (Default = 1000)", type = int, default = 1000)
parser.add_argument("-attack", help = "Foolbox = 0; Custom PGD = 1, Min PGD = 2, Fast DDN = 3", type = int, default = 0)
parser.add_argument("-restarts", help = "Default = 10", type = int, default = 10)
parser.add_argument("-path", help = "To override default model fetching- Automatically appends '.pt' to path", type = str)
parser.add_argument("-subset", help = "Subset of attacks", type = int, default = -1)
parser.add_argument("-epsilon_l_1", help = "Eps", type = float, default = 12)
parser.add_argument("-epsilon_l_2", help = "Eps", type = float, default = 0.5)
parser.add_argument("-epsilon_l_inf", help = "Eps", type = float, default = 0.031)
parser.add_argument("-alpha_l_1", help = "Eps", type = float, default = 1.0)
parser.add_argument("-alpha_l_2", help = "Eps", type = float, default = 0.01)
parser.add_argument("-alpha_l_inf", help = "Eps", type = float, default = 0.001)


params = parser.parse_args()

device_id = params.gpu_id
batch_size = params.batch_size
choice = params.model
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


def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'SAPA':
        A = fa.SaltAndPepperNoiseAttack()
    elif attack == 'PA':
        A = fa.L1BrendelBethgeAttack()

    # L2
    elif 'IGD' in attack:
        A = fa.L2BasicIterativeAttack()
    elif attack == 'AGNA':
        A = fa.L2AdditiveGaussianNoiseAttack()
    elif attack == 'BA':
        A = fa.BoundaryAttack()
    elif 'DeepFool' in attack:
        A = fa.L2DeepFoolAttack()
    elif attack == 'PAL2':
        A = fa.L2BrendelBethgeAttack()
    elif attack == "CWL2":
        A = fa.L2CarliniWagnerAttack()


    # L inf
    elif 'FGSM' in attack and not 'IFGSM' in attack:
        A = fa.FGSM()
    elif 'PGD' in attack:
        A = fa.LinfPGD()
    elif 'IGM' in attack:
        A = fa.LinfBrendelBethgeAttack()
    else:
        raise Exception('Not implemented')
    return A, 0,0,0


def parse_subset(subset):
    if subset == 0:
        attacks_list = ['PA','SAPA']
        types_list   = [ 0    , 0 ]
    elif subset == 1:
        types_list   = [ 2  ]
        attacks_list = ['BA']
        max_tests = 100
    elif subset == 2:
        attacks_list = ['IGD','AGNA','DeepFool','PAL2']
        types_list = [2,2,2,2]
    elif subset == 3 :
        attacks_list =['PGD','FGSM','IGM','CWL2']
        types_list = [3,3,3,2]
    else:
        attacks_list = ['SAPA','PA','IGD','AGNA','BA','DeepFool','PAL2','CWL2','FGSM','PGD','IGM']
        types_list   = [ 0    , 0  , 2   , 2    ,  2  ,  2  ,      2    , 2,   3      , 3   , 3 ]
    return attacks_list, types_list    

def test_foolbox(model, model_name, max_check = 1000):
    #Saves the minimum epsilon value for successfully attacking each image via different foolbox attacks as an npy file in the folder corresponding to model_name
    #No Restarts in case of BA
    #Batch size = 1 is supported
    torch.manual_seed(0)
    model = model.eval()
    preprocessing = dict()
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing, device = device)
    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1)
    

    attacks_list, types_list = parse_subset(subset)
    norm_dict = {0:norms_l0_squeezed, 1:norms_l1_squeezed, 2:norms_l2_squeezed,3:norms_linf}
    epsilons_dict = {0:10, 1:10, 2:0.5, 3:8/255.}

    for i in range(len(attacks_list)):
        # ipdb.set_trace()
        attack_name = attacks_list[i]
        restarts = 1 if attack_name == "BA" else res
        print (attack_name)
        types = types_list[i]
        norm = norm_dict[types]
        train_loader, test_loader = get_dataloaders(batch_size)

        start = time()
        output = np.ones((max_check))
        
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        for images,labels in test_loader:        
            #ipdb.set_trace()
            images = images.to(device)
            labels = labels.to(device)
            start = time()
            # ipdb.set_trace()
            criterion = fb.criteria.Misclassification(labels)
            distance = 1000*torch.ones(batch_size).half().to(device)
            advs_final = images
            for r in range (restarts):
                advs, clipped, is_adv = attack(fmodel, images, criterion, epsilons=epsilons_dict[types])
                new_distance = norm(images-advs)
                advs_final[distance > new_distance] = advs[distance>new_distance]
                distance[distance > new_distance] = new_distance[distance>new_distance]
                
                
            classes = model(advs_final)
            num_correct = classes.max(1)[1] == labels
            output[total:total+batch_size] = distance.cpu()
            total += batch_size
            print(total, " ", attack_name, " " ,model_name, " distance = ", distance.mean(), " num correct = ", num_correct.sum().item(), " Time taken = ", time() - start,)
            
            if (total >= max_check):
                np.save(model_name + "/" + attack_name + ".npy" ,output)
                break

        print("Time Taken = ", time() - start)



def test_pgd(model, model_name, clean = False):
    #Computes the adversarial accuracy at standard thresholds of (0.3,1.5,12) for first 1000 images

    print (model_name)
    print(device)

    lr_schedule = None
    epoch_i = 0
    
    # total_loss, total_acc = epoch(test_loader, lr_schedule, model, epoch_i, criterion, opt = None, device = device, stop = True)
    # print('Test Acc Clean: {0:.4f}'.format(total_acc))
    # # total_loss, total_acc_inf = epoch_adversarial(test_loader, None, model, epoch_i, attack_pgd, device = device, stop = True, restarts = res)
    # total_loss, total_acc_inf = epoch_adversarial(test_loader, None, model, epoch_i, pgd_linf, 
    #                         device = device, stop = True, restarts = res, 
    #                         epsilon = params.epsilon_l_inf, alpha = params.alpha_l_inf)
    # print('Test Acc Inf: {0:.4f}'.format(total_acc_inf))    
    total_loss, total_acc_1 = epoch_adversarial(test_loader, None,  model, epoch_i, pgd_l1_topk, 
                            device = device, stop = True, restarts = res,
                            epsilon = params.epsilon_l_1, alpha = params.alpha_l_1, num_iter =10)
    print('Test Acc 1: {0:.4f}'.format(total_acc_1))    
    # total_loss, total_acc_2 = epoch_adversarial(test_loader, None, model, epoch_i, pgd_l2, 
    #                         device = device, stop = True, restarts = res,
    #                         epsilon = params.epsilon_l_2, alpha = params.alpha_l_2)
    # print('Test Acc 2: {0:.4f}'.format(total_acc_2))    



def fast_adversarial_DDN(model_name):
    #Saves the minimum epsilon value for successfully attacking each image via PGD based attack as an npy file in the folder corresponding to model_name
    #No Restarts
    #Done for a single batch only since batch size is supposed to be set to 1000 (first 1000 images)
    print (model_name)
    print(device)
    model = PreActResNet18().to(device)
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half()   

    model.load_state_dict(torch.load(model_name+".pt", map_location = device))
    model.eval()
    
    for i,batch in enumerate(test_loader): 
        x,y = batch[0].to(device), batch[1].to(device)
        attacker = DDN(steps=100, device=device)
        adv = attacker.attack(model, x, labels=y, targeted=False)
        delta = (adv - x)
        norm = norms(delta).squeeze(1).squeeze(1).squeeze(1).cpu().numpy() 
        print('Test Acc L2: {0:.4f}'.format(norm[norm<0.5].sum()/norm.shape[0]))    
        np.save(model_name + "/" + "DDN" + ".npy" ,norm) 
        break



def test_saver(model_name):
    #Saves the minimum epsilon value for successfully attacking each image via PGD based attack as an npy file in the folder corresponding to model_name
    eps_1 = [3,6,(2000/255),12,20,30,50,60,70,80,90,100]
    eps_2 = [0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
    eps_3 = [0.005,0.01,(4/255),0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
    num_1 = [50,50,100,100,100,200,200,200,300,300,300,300]
    num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
    num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]

    attacks_l1 = torch.ones((batch_size, 12))*1000
    attacks_l2 = torch.ones((batch_size, 12))*1000
    attacks_linf = torch.ones((batch_size, 12))*1000
    model = PreActResNet18().cuda()
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half()   
            
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    criterion = nn.CrossEntropyLoss()

    model.eval()        
    test_loader = Batches(test_set, batch_size, shuffle=False, gpu_id = device_id)

    try:
        total_loss, total_acc = epoch(test_loader, None, model, 0, criterion, opt = None, device = device, stop = True)
    except:
        print ("OK")

    for index in range(len(eps_1)):
            e_1 = eps_1[index]
            n_1 = num_1[index]
            eps, total_acc_1 = epoch_adversarial_saver(batch_size, test_loader, model, pgd_l1, e_1, n_1, device = device, restarts = res)
            attacks_l1[:,index] = eps
    attacks_l1 = torch.min(attacks_l1,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL1" + ".npy" ,attacks_l1.numpy())

    for index in range(len(eps_2)):        
            e_2 = eps_2[index]
            n_2 = num_2[index]
            eps, total_acc_2 = epoch_adversarial_saver(batch_size, test_loader, model, pgd_l2, e_2, n_2, device = device, restarts = res)
            attacks_l2[:,index] = eps
    attacks_l2 = torch.min(attacks_l2,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL2" + ".npy" ,attacks_l2.numpy())

    for index in range(len(eps_3)):
            e_3 = eps_3[index]
            n_3 = num_3[index]
            eps, total_acc_3 = epoch_adversarial_saver(batch_size, test_loader, model, pgd_linf, e_3, n_3, device = device, restarts = res)
            attacks_linf[:,index] = eps
    attacks_linf = torch.min(attacks_linf,dim = 1)[0]
    np.save(model_name + "/" + "CPGDLINF" + ".npy" ,attacks_linf.numpy())
  


model_list = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA"]
model_name = "Selected/{}".format(model_list[choice])
if path is not None:
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
    test_foolbox(model,model_name, 1000)
elif attack == 1:
    test_pgd(model,model_name)
elif attack ==2:
    test_saver(model_name)
elif attack == 3:
    fast_adversarial_DDN(model_name)