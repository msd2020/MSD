## Pretrained Models  
Pretrained models for each of the training methods discussed in the paper are available at the link:
`https://drive.google.com/open?id=1EPMYz5VqjhhJaxcpUgAiZvqS3bYbnjHM`
Please download these model in a folder named `Selected`  to test the models.
The testing code is automatically designed to pick the models from that folder. 

## Testing Code

+ `test.py` - Test the Adversarially Robust Models
  > `gpu_id`  - Id of GPU to be used  - `default = 0`  
  > `model`   - Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: avg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: max  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Test Set -`default = 100`  
  > `attack` - Foolbox = 0; Custom PGD = 1, Min PGD = 2, Fast DDN = 3;  - `default = 1`  
  > `restarts`  - Number of Random Restarts - `default = 10`  
  > `path` - To override default model fetching - `default = None`   
  > `subset` - Subset for Foolbox attacks - `default = -1`   


## Training Code

+ `train.py` - Train the Adversarially Robust Models
  > `gpu_id`  - Id of GPU to be used  - `default = 0`  
  > `model`   - Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: avg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: max  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Train Set -`default = 128` 


## For Reproducing the Results


**0. P_inf:**  
`python train.py -model 0`
  
**1. P1:**  
`python train.py -model 1`

  Trained with k = 1 for pgd_l1_topk (needs to be set in cifar_funcs.py)
  Early Stop: Epoch 45

**2. P2:**  
`python train.py -model 2`
  
**3. MSD:**   
`python train.py -model 3`
  
**4. AVG:**  
`python train.py -model 4`

**5. MAX:**  
`python train.py -model 5`

  alpha_inf = 0.005 
  Early stop : Epoch 40
