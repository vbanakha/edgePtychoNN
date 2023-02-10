import torch
import torch.nn as nn
#import onnx
from torchsummary import summary

import sys

import numpy as np

class ReconSmallPhaseModel(nn.Module):
    def __init__(self, nconv: int = 16):
        super(ReconSmallPhaseModel, self).__init__()
        self.nconv = nconv
        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8), 
            *self.down_block(self.nconv * 8, self.nconv * 16), 
            *self.down_block(self.nconv * 16, self.nconv * 32),
            *self.down_block(self.nconv * 32, self.nconv * 32)
        )
        
        # amplitude model
        #self.decoder1 = nn.Sequential(
            #*self.up_block(self.nconv * 32, self.nconv * 32),
         #   *self.up_block(self.nconv * 32, self.nconv * 16),
          #  *self.up_block(self.nconv * 16, self.nconv * 8),
           # *self.up_block(self.nconv * 8, self.nconv * 8),
            #*self.up_block(self.nconv * 8, self.nconv * 4),
            #*self.up_block(self.nconv * 4, self.nconv * 2),
            #*self.up_block(self.nconv * 2, self.nconv * 1),
            #nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        #)
        
        # phase model
        self.decoder2 = nn.Sequential(
            #*self.up_block(self.nconv * 32, self.nconv * 32),
            *self.up_block(self.nconv * 32, self.nconv * 16),
            *self.up_block(self.nconv * 16, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
            nn.Tanh()
        )
    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ]
        return block
    
    
    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]
        return block
        
    
    def forward(self,x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amp = self.decoder1(x1)
            ph = self.decoder2(x1)
            #Restore -pi to pi range
            ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi
        return ph

def main():
    newFilePath = "best_model.pth" # specify the pth trained file here 
    model = ReconSmallPhaseModel()
    state_dict = torch.load(newFilePath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    summary(model, (1, 512, 512))

    dummy_input = (torch.randn(bsz, 1, 512, 512)) # batchsize , 1, h, w
    torch.onnx.export(model, dummy_input, "ptychoNN_{0}.onnx".format(bsz), opset_version=12)
if __name__ == '__main__':
    bsz = int(sys.argv[1])
    main()


