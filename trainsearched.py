import os
import time
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
from nas_bench_201_search import SNASNet
from spikingjelly.clock_driven.functional import reset_net
import torch.nn.functional as F
from preprocessing import NeuroTacDataset



def main():
    print(torch.__version__)
    print(hasattr(torch.serialization, 'safe_globals'))

    with torch.serialization.safe_globals([NeuroTacDataset]):
        #load data
        batchsize = 32
        
        dataset = torch.load("/Users/lanceshi/school/NeuroTacDataSetUInt8_cropped.pth", weights_only=False)
        
        trainset, valset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True,
                                                    num_workers=4, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False,
                                                    num_workers=4, pin_memory=True)


        best_neuroncell = torch.tensor([[0, 3, 0, 2],
                                        [0, 0, 2, 3],
                                        [0, 0, 0, 4],
                                        [0, 0, 0, 0]])
        save_model_path = "./savemodel/"
        epochs = 300
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 5e-4
        val_interval = 1
        
        
        model = SNASNet(best_neuroncell).to(torch.device("mps"))
        criterion = nn.CrossEntropyLoss().to(torch.device("mps"))
    
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= epochs, eta_min= learning_rate * 0.01)
        
        start = time.time()
        for epoch in range(epochs):
            train(epochs, epoch, train_loader, model, criterion, optimizer, scheduler)
            scheduler.step()
            #if (epoch + 1) % val_interval == 0:
            #    validate(args, epoch, val_loader, model, criterion)
            #    utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args.exp_name + '_super')
        #utils.time_record(start)
    
    
    

def train(epochs, this_epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    #top1 = utils.AvgrageMeter()
    if (this_epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', this_epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))

    for step, (inputs, target) in enumerate(train_data):
        print(target)
        inputs = inputs.to(torch.device("mps"))
        target = F.one_hot(target)
        target.to(torch.device("mps"))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = F.mse_loss(outputs, target)
        loss.backward()
        
        optimizer.step()
        
        #prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        #n = inputs.size(0)
        #top1.update(prec1.item(), n)
        train_loss += loss.item()
        print(train_loss)
        reset_net(model)
    print('train_loss: %.6f' % (train_loss / len(train_data)))


def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(torch.device("mps")), targets.to(torch.device("mps"))
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            reset_net(model)
        print('[Val_Accuracy epoch:%d] val_acc:%f'
              % (epoch + 1,  val_top1.avg))
        return val_top1.avg
  
if __name__ == '__main__':
    main()