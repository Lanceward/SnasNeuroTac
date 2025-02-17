import torch
import torch.nn as nn
import numpy as np
#from spikingjelly.clock_driven import functional, surrogate, neuron
from spikingjelly.activation_based import layer, neuron, surrogate
import matplotlib.pyplot as plt
from preprocessing import NeuroTacDataset
import gc

def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return  ld

def find_best_neuroncell(trainset):

    search_batchsize = 64
    repeat = 2
    num_search = 500
    timestep = 1000
    group = 10
    
    #trainig data manipulation
    train_data = torch.utils.data.DataLoader(trainset, batch_size=search_batchsize,
                                               shuffle=True, pin_memory=False, num_workers=2)
    
    scores = []
    history = []
    neuron_type = 'LIFNode'

    with torch.no_grad():
        for i in range(num_search):
            #generate random hidden_neurons 
            #hidn = torch.randint(16, 512, (1,))
            hidn = torch.randint(100, 900, (1,))
            
            searchnet = DenseSpikingNN(hidn.item(), timestep)
            #searchnet = ConvSpikingNN(hidn, timestep)
            searchnet.to(torch.device("mps"))
            
            searchnet.K = np.zeros((search_batchsize, search_batchsize))
            searchnet.num_actfun = 0
            
            def computing_K_eachtime(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.view(out.size(0), -1)
                batch_num , neuron_num = out.size()
                x = (out > 0).float()

                full_matrix = torch.ones((search_batchsize, search_batchsize), device="mps") * neuron_num
                sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
                norm_K = ((sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t()))
                rescale_factor = torch.div(0.5 * torch.ones((search_batchsize, search_batchsize), device="mps"), norm_K+1e-3)
                K1_0 = (x @ (1 - x.t()))
                K0_1 = ((1-x) @ x.t())
                K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))
                
                searchnet.K = searchnet.K + (K_total.cpu().numpy())
                searchnet.num_actfun += 1

            s = []
            #for module in searchnet.conv_fc:
            for name, module in searchnet.named_modules():
                if neuron_type in str(type(module)):
                    module.register_forward_hook(computing_K_eachtime)

            for j in range(repeat):
                searchnet.K = np.zeros((search_batchsize, search_batchsize))
                searchnet.num_actfun = 0
                data_iterator = iter(train_data)
                inputs, targets = next(data_iterator)
                inputs, targets = inputs.to(torch.float32), inputs.to(torch.float32)
                inputs, targets = inputs.to(torch.device("mps")), targets.to(torch.device("mps"))
                outputs = searchnet(inputs)
                #print(searchnet.K/ (searchnet.num_actfun))
                score = logdet(searchnet.K/ (searchnet.num_actfun))
                print("" + str(hidn.item()) + ", " + str(score))
                s.append(score)

            scores.append(np.mean(s))
            history.append(hidn)
            gc.collect()
        
        print(scores)
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
        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 1.33)  
        
        # Fully connected layer: hidden_neurons -> 11 (output)
        self.fc2 = nn.Linear(hidden_neurons, 11)
        # Spiking neuron layer after the second FC layer
        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 1.33)  
        
        self.timestep = timestep
        
        nn.init.uniform_(self.fc1.weight, a = 0, b = 1)
        nn.init.uniform_(self.fc2.weight, a = 0, b = 1)
        
    def forward(self, x):
        # Assume x is of shape [batchsize, T, 2, 40, 40]
        # transform x into [T, batchsize, 1600]
        x = x[:, :, 1, :, :]
        #print(torch.sum(x))
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = torch.swapaxes(x, 0, 1)
        
        #print(x.shape)
        
        for t in range(self.timestep):
            x_single = x[t]
            x_single = self.fc1(x_single)
            x_single = self.lif1(x_single)
            x_single = self.fc2(x_single)
            x_single = self.lif2(x_single)
        return x_single

class ConvSpikingNN(nn.Module):
    def __init__(self, channels: int, timestep: int):
        super(ConvSpikingNN, self).__init__()
        
        self.timestep = timestep
        
        self.conv_fc = nn.Sequential(
        layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, tau= 1.33),
        layer.MaxPool2d(2, 2),  # 20 * 20

        layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, tau= 1.33),
        layer.MaxPool2d(2, 2),  # 10 * 10

        layer.Flatten(),
        layer.Linear(channels * 10 * 10, channels * 4 * 4, bias=False),
        neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, tau= 1.33),

        layer.Linear(channels * 4 * 4, 11, bias=False),
        neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, tau= 1.33),
        )
        
        for m in self.conv_fc:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a =2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor):
        # Assume x is of shape [batchsize, T, 2, 40, 40]
        # transform x into [T, batchsize, 40, 40]
        x = x[:, :, 1, :, :]
        x = torch.unsqueeze(x, dim= 2)
        x = torch.swapaxes(x, 0, 1)
                
        for t in range(self.timestep):
            x_single = x[t]
            x_single = self.conv_fc(x_single)
        return x_single

if __name__ == "__main__":
    with torch.serialization.safe_globals([NeuroTacDataset]):
        ds = torch.load("/Users/lance-shi/school/NeuroTacDataSetUInt8_cropped.pth", weights_only=False)
        
        best_policy = find_best_neuroncell(ds)