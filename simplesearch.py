import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
import matplotlib.pyplot as plt


def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return  ld

def find_best_neuroncell(trainset):

    search_batchsize = 256
    repeat = 2
    num_search = 5000
    timestep = 5
    #parameter_limit = 5 * 1000 * 1000 # 5 million parameters
    
    #trainig data manipulation
    #pass
    
    scores = []
    history = []
    neuron_type = 'LIFNode'

    with torch.no_grad():
        for i in range(num_search):
            #generate random hidden_neurons 
            hidn = torch.randint(200, 900)
            
            searchnet = DenseSpikingNN(hidn)
            searchnet.K = np.zeros((search_batchsize, search_batchsize))
            searchnet.num_actfun = 0
            
            def computing_K_eachtime(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.view(out.size(0), -1)
                batch_num , neuron_num = out.size()
                x = (out > 0).float()

                #full_matrix = torch.ones((search_batchsize, search_batchsize)).cuda() * neuron_num
                full_matrix = torch.ones((search_batchsize, search_batchsize)) * neuron_num
                sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
                norm_K = ((sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t())) * neuron_num
                #rescale_factor = torch.div(0.5* torch.ones((search_batchsize, search_batchsize)).cuda(), norm_K+1e-3)
                rescale_factor = torch.div(0.5* torch.ones((search_batchsize, search_batchsize)), norm_K+1e-3)
                K1_0 = (x @ (1 - x.t()))
                K0_1 = ((1-x) @ x.t())
                K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))

                searchnet.K = searchnet.K + (K_total.cpu().numpy())
                searchnet.num_actfun += 1

            s = []
            for name, module in searchnet.named_modules():
                if neuron_type in str(type(module)):
                    module.register_forward_hook(computing_K_eachtime)

            for j in range(repeat):
                searchnet.K = np.zeros((search_batchsize, search_batchsize))
                searchnet.num_actfun = 0
                data_iterator = iter(train_data)
                inputs, targets = next(data_iterator)
                #inputs, targets = inputs.cuda(), targets.cuda()
                outputs = searchnet(inputs)
                score = logdet(searchnet.K/ (searchnet.num_actfun))
                print("Hidden layer: " + str(hidn) + "Score: " + str(score))
                s.append(score)

            scores.append(np.mean(s))
            history.append(hidn)
        
        print ("mean / var:", np.mean(scores), np.var(scores))
        print ("max score:", max(scores))
        best_idx = (np.argsort(scores))[-1]
        best_policy = history[best_idx]
        print ("best policy: " + str(best_policy))
        print (scores)
        print (history)
        
    return best_policy
        
class DenseSpikingNN(nn.Module):
    def __init__(self, hidden_neurons: int, timestep: int):
        super(DenseSpikingNN, self).__init__()
        # Fully connected layer: Input 1600 -> hidden_neurons
        self.fc1 = nn.Linear(1600, hidden_neurons)
        # Spiking neuron layer after the first FC layer
        self.lif1 = neuron.LIFNode()  
        
        # Fully connected layer: hidden_neurons -> 11 (output)
        self.fc2 = nn.Linear(hidden_neurons, 11)
        # Spiking neuron layer after the second FC layer
        self.lif2 = neuron.LIFNode()  
        
        self.timestep = timestep
        
    def forward(self, x):
        # Assume x is of shape [batch_size, 1600]
        for t in range(self.timestep):
            x_single = x[t]
            x_single = self.fc1(x_single)
            x_single = self.lif1(x_single)
            x_single = self.fc2(x_single)
            x_single = self.lif2(x_single)
        return x_single