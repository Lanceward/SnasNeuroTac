import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from searchcells.search_cell_snn import Neuronal_Cell,Neuronal_Cell_backward

"""
this searchspace is similar to NAS-BENCH-201
copied from Kim's work. Do not use backward cells
"""

class SNASNet(nn.Module):
    def __init__(self, args, con_mat):
        super(SNASNet, self).__init__()

        self.con_mat = con_mat
        self.total_timestep = args.timestep
        self.second_avgpooling = args.second_avgpooling

        # for a 1000x40x40 input, from Bristol/SNN/SlideS10
        # copy mostly cifar10 settings
        self.num_class = 10
        self.num_final_neuron = 100
        self.num_cluster = 10
        self.in_channel = 1
        self.img_size = 40
        self.first_out_channel = 16#128
        self.spatial_decay = 2 * self.second_avgpooling
        self.classifier_inter_ch = 512#1024
        self.stem_stride = 1
        # settings finished

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channel, self.first_out_channel, kernel_size=3, stride=self.stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channel, affine=True),
        )

        self.cell1 = Neuronal_Cell(args, self.first_out_channel, self.first_out_channel, self.con_mat)

        self.downconv1 = nn.Sequential(
            nn.BatchNorm2d(self.first_out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau= args.tau,
                                                      surrogate_function=surrogate.ATan(),
                                                      detach_reset=True),
                                        nn.Conv2d(self.first_out_channel, self.first_out_channel*2, kernel_size=(3, 3),
                                                  stride=(1, 1), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(self.first_out_channel*2, eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True)
                                        )
        self.resdownsample1 = nn.AvgPool2d(2,2)

        self.cell2 = Neuronal_Cell(args, self.first_out_channel*2, self.first_out_channel*2, self.con_mat)

        self.last_act = nn.Sequential(
                        nn.BatchNorm2d(self.first_out_channel*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True)
        )
        self.resdownsample2 = nn.AvgPool2d(self.second_avgpooling,self.second_avgpooling)

        self.classifier = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(self.first_out_channel*2*(self.img_size//self.spatial_decay)*(self.img_size//self.spatial_decay), self.classifier_inter_ch, bias=False),
            neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
        nn.Linear(self.classifier_inter_ch, self.num_final_neuron, bias=True))
        self.boost = nn.AvgPool1d(self.num_cluster, self.num_cluster)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a =2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        acc_voltage = 0
        batch_size = input.size(0)

        # transform input into [T, batchsize, 1, 40, 40]
        input = input[:, :, 1, :, :]
        input = torch.unsqueeze(input, dim = 2)
        input = torch.swapaxes(input, 0, 1)

        for t in range(self.total_timestep):
            x = input[t]
            x = self.stem(x)
            x = self.cell1(x)
            x = self.downconv1(x)
            x = self.resdownsample1(x)
            x = self.cell2(x)
            x = self.last_act(x)
            x = self.resdownsample2(x)
            x = x.view(batch_size, -1)
            x = self.classifier(x)
            acc_voltage = acc_voltage + self.boost(x.unsqueeze(1)).squeeze(1)
        acc_voltage = acc_voltage / self.total_timestep
        return acc_voltage

