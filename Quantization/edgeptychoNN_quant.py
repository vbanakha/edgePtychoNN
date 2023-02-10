import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
import os 
from torchsummary import summary

""" from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True """

## not able to export to ONNX when performing the quantization in pytorch 
## trying out the export in pytorch-quantization

class ReconSmallPhaseModel(nn.Module):
    def __init__(self, nconv: int = 32,  quantize=True):
        super(ReconSmallPhaseModel, self).__init__()
        self.quantize = quantize
        self.nconv = nconv
        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8), 
            *self.down_block(self.nconv * 8, self.nconv * 16),
            *self.down_block(self.nconv * 16, self.nconv * 32)
        )
        
        # amplitude model
        #self.decoder1 = nn.Sequential(
         #   *self.up_block(self.nconv * 32, self.nconv * 32),
         #   *self.up_block(self.nconv * 32, self.nconv * 16),
          #  *self.up_block(self.nconv * 16, self.nconv * 8),
         #   *self.up_block(self.nconv * 8, self.nconv * 8),
         #   *self.up_block(self.nconv * 8, self.nconv * 4),
        #   *self.up_block(self.nconv * 4, self.nconv * 2),
         #   *self.up_block(self.nconv * 2, self.nconv * 1),
        #   nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        #)
        
        # phase model
        self.decoder2 = nn.Sequential(
            *self.up_block(self.nconv * 32, self.nconv * 16), #16
            *self.up_block(self.nconv * 16, self.nconv * 8),#32
            *self.up_block(self.nconv * 8, self.nconv * 4),#64
            *self.up_block(self.nconv * 4, self.nconv * 2),#128
            #*self.up_block(self.nconv * 2, self.nconv * 1),
            #*self.up_block(self.nconv * 1, 16),
            nn.Conv2d(self.nconv*2 , 1, 3, stride=1, padding=(1,1)),
            nn.Tanh()
        )

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
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
        if self.quantize:
            x = self.quant(x)
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amp = self.decoder1(x1)
            ph = self.decoder2(x1)
            #Restore -pi to pi range
            ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi
        if self.quantize:
            ph = self.dequant(ph)    
        return ph


def main():
    newFilePath = "/home/beams/ABABU/ptychoNN-test/models_11_22/best_model_reduced_model.pth"
    model_org = ReconSmallPhaseModel()
    dummy_input = (torch.randn(1, 1, 512, 512))

    state_dict = torch.load(newFilePath, map_location=torch.device('cpu'))
    model_org.load_state_dict(state_dict)
    summary(model_org, (1, 512, 512))

    model_org.to('cpu')
    
    model_org.eval()
    print(model_org) ## can get the layer names from this, use that in modules_to_fuse []

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
                   ['decoder2.0', 'decoder2.1'],  
                   ['decoder2.2', 'decoder2.3'], 
                   ['decoder2.5', 'decoder2.6'],
                   ['decoder2.7', 'decoder2.8'], 
                   ['decoder2.10', 'decoder2.11'], 
                   ['decoder2.12', 'decoder2.13'], 
                   ['decoder2.15', 'decoder2.16'],
                   ['decoder2.17', 'decoder2.18']] 

    model_org.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    model_fused = torch.quantization.fuse_modules(model_org, modules_to_fuse)
    model_prepared = torch.quantization.prepare(model_fused)

    model_prepared = torch.quantization.prepare(model_fused)

    model_prepared(dummy_input)
    model_int8 = torch.quantization.convert(model_prepared)
    #output_check = model_int8(dummy_input)
    print_size_of_model(model_org)
    print("========================================= PERFORMANCE =============================================")
    print_size_of_model(model_int8)
    print("========================================= PERFORMANCE =============================================")


    ## load the quantized pth file and export to onnx using pytorch-quantization (not working) 
    #quant_modules.initialize()
    #state_dict = torch.load("temp.pth", map_location="cpu")
    #model_int8.load_state_dict(state_dict)
    #dummy_input = torch.randn(1, 1, 512, 512, device='cpu')

    #torch.onnx.export(
    #model_int8, dummy_input, "quant_ptychoNN.onnx", verbose=True, opset_version=10)

    """
    Saving a quantized model in onnx format 
    """
    """ torch.onnx.export(model_int8,             # model being run
                    dummy_input,                         # model input (or a tuple for multiple inputs)
                    'model_int8.onnx',   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,        # the ONNX version to export the model to
                    #enable_onnx_checker=False
                    #do_constant_folding=True,  # whether to execute constant folding for optimization
                    #input_names = ['input'],   # the model's input names
                    #output_names = ['output'], # the model's output names
                    #example_outputs=traced(input_fp32)
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                    verbose = True
                    )
 """                    
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
