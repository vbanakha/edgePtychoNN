import numpy as np
import threading

from helper import inference

class inferPtychoNNtrt:
    def __init__(self, pvapyProcessor, mbsz, onnx_mdl, tq_diff , frm_id_q):
        self.tq_diff = tq_diff
        self.mbsz = mbsz
        self.onnx_mdl = onnx_mdl
        self.pvapyProcessor= pvapyProcessor
        self.frm_id_q = frm_id_q
        import tensorrt as trt
        from helper import engine_build_from_onnx, mem_allocation, inference
        import pycuda.autoinit # must be in the same thread as the actual cuda execution
        self.context = pycuda.autoinit.context
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, \
            self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()

    def stop(self):
        try:
            self.context.pop()
        except Exception as ex:
            pass

    def batch_infer(self, nx, ny):
        in_mb  = self.tq_diff.get()
        bsz, ny, nx = in_mb.shape
        frm_id_list = self.frm_id_q.get()
        np.copyto(self.trt_hin, in_mb.astype(np.float32).ravel())
        pred = np.array(inference(self.trt_context, self.trt_hin, self.trt_hout, \
                             self.trt_din, self.trt_dout, self.trt_stream))
        
        pred = pred.reshape(bsz, nx*ny)    
        for j in range(0, len(frm_id_list)):  
            image = pred[j].reshape(ny,nx)
            frameId = int(frm_id_list[j])
            outputNtNdArray = self.pvapyProcessor.generateNtNdArray2D(frameId, image)
            self.pvapyProcessor.updateOutputChannel(outputNtNdArray)
