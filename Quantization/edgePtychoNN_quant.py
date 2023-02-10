import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
import os 
from torchsummary import summary

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True

import time 
## not able to export to ONNX when performing the quantization in pytorch 
## trying out the export in pytorch-quantization

class ReconSmallPhaseModel(nn.Module):
    def __init__(self, nconv: int = 16, quantize=True):
        super(ReconSmallPhaseModel, self).__init__()
        self.nconv = nconv
        self.quantize = quantize
        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8), 
            *self.down_block(self.nconv * 8, self.nconv * 16),
            *self.down_block(self.nconv * 16, self.nconv * 16),
        )
        
        # amplitude model
        #self.decoder1 = nn.Sequential(
        #    *self.up_block(self.nconv * 8, self.nconv * 8),
        #    *self.up_block(self.nconv * 8, self.nconv * 4),
        #    *self.up_block(self.nconv * 4, self.nconv * 2),
        #    *self.up_block(self.nconv * 2, self.nconv * 1),
        #    nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        #)
        
        # phase model
        self.decoder2 = nn.Sequential(
            *self.up_block(self.nconv * 16, self.nconv * 16),
            *self.up_block(self.nconv * 16, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            #*self.up_block(self.nconv * 4, self.nconv * 2),
            #*self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 4, 1, 3, stride=1, padding=(1,1)),
            nn.Tanh()
        )

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.mult_scalar = nn.quantized.FloatFunctional()
        

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=False),
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
        
        x = self.quant(x)
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amp = self.decoder1(x1)
            ph = self.decoder2(x1)

            #Restore -pi to pi range
            ph = self.dequant(ph)
            #ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi
            ph = ph*np.pi

        #if self.quantize:
          
        return ph


def main():
    newFilePath = "/home/beams/ABABU/ptychoNN-test/models_02_23/best_model_reduced_model.pth"
    model = ReconSmallPhaseModel()
    dummy_input = (torch.randn(1, 1, 512, 512))

    state_dict = torch.load(newFilePath, map_location=torch.device('cpu'))
    
    model.load_state_dict(state_dict)
    summary(model, (1, 512, 512))

    model.to('cpu')
    
    model.eval()
    print(model) ## can get the layer names from this, use that in modules_to_fuse []

    modules_to_fuse = [['encoder.0', 'encoder.1'],
                   ['encoder.2', 'encoder.3'],
                   ['encoder.5', 'encoder.6'],
                   ['encoder.7', 'encoder.8'],
                   ['encoder.10', 'encoder.11'],
                   ['encoder.12', 'encoder.13'],
                   ['encoder.15', 'encoder.16'],
                   ['encoder.17', 'encoder.18'],
                   ['encoder.20', 'encoder.21'],
                   ['encoder.22', 'encoder.23'],
                   ['encoder.25', 'encoder.26'],
                   ['encoder.27', 'encoder.28'],
                   ['decoder2.0', 'decoder2.1'],  
                   ['decoder2.2', 'decoder2.3'], 
                   ['decoder2.5', 'decoder2.6'],
                   ['decoder2.7', 'decoder2.8'], 
                   ['decoder2.10', 'decoder2.11'], 
                   ['decoder2.12', 'decoder2.13'], 
                   ['decoder2.15', 'decoder2.16'],
                   ['decoder2.17', 'decoder2.18']]  

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    model_fused = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    model_prepared = torch.quantization.prepare(model_fused, inplace=True)

    model_prepared = torch.quantization.prepare(model_fused)

    model_prepared(dummy_input)
    model_int8 = torch.quantization.convert(model_prepared, inplace=True)
    
    print_size_of_model(model)
    print("========================================= PERFORMANCE =============================================")
    print_size_of_model(model_int8)
    print("========================================= PERFORMANCE =============================================")


    model_scripted = torch.jit.script(model_int8) # Export to TorchScript
    model_scripted.save('model_scripted_old.pt')




   
    ## load the quantized pth file and export to onnx using pytorch-quantization (not working) 
    
    ## test the inference 
    device = torch.device("cuda")
    model_int8.to(device)
    model_int8.eval()
    
    dummy_input.cuda()
#model_in = X_test[0].reshape(1, 1, 64, 64)
#model_in1 = torch.Tensor(dummy_input).to(device)

    

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))
#GPU-WARM-UP
    for _ in range(10):
        _ = model_int8(dummy_input)
# MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            st = time.time()
            model_out= model_int8(dummy_input)
            et = time.time()
        # WAIT FOR GPU SYNC
            #torch.cuda.synchronize()
            #curr_time = starter.elapsed_time(ender)
            timings[rep] = et-st
    
#np.save('model_out_0.npy', (model_out[0].detach().to("cpu").numpy()))
#np.save('model_out_1.npy', (model_out[1].detach().to("cpu").numpy()))
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("Inference time for a batch size of 8 is %f s" %mean_syn)
    print("Standard Deviation is %f s "%std_syn)

    #torch.onnx.export(
    #model_int8, dummy_input, "quant_ptychoNN.onnx", verbose=True, opset_version=10) 

    
   
                     
def print_size_of_model(model):
    """ Print the size of the model.
    
    Args:
        model: model whose size needs to be determined

    """
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')





if __name__ == '__main__':
    main()           
