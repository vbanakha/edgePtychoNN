from PIL import Image
import numpy as np
import tensorrt as trt
#import pycuda.autoinit
#import pycuda.driver as cuda
import threading
import time
import math
import os 
import logging
#import GPUtil
#import common_v1

from multiprocessing import Process, Queue
from skimage.transform import resize
from helper import inference
from pvaClient import * 

class inferPtychoNNtrt:
    def __init__(self, client, mbsz, onnx_mdl, tq_diff , frm_id_q):
        self.tq_diff = tq_diff
        self.mbsz = mbsz
        self.onnx_mdl = onnx_mdl
        self.client= client
        self.frm_id_q = frm_id_q
        self.processed_count = 0 
        self.msg1 = ''
        self.msg2 = ''
        self.frame_loss = 0
        self.t0=0
        from helper import engine_build_from_onnx, mem_allocation, inference
        import pycuda.autoinit # must be in the same thread as the actual cuda execution
        
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, \
            self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()
        logging.info("TensorRT Inference engine initialization completed!")

    def start(self, ):
        threading.Thread(target=self.batch_infer, daemon=True).start()
        

    def batch_infer(self, ):

        

        ## change here, tensorrt engine need not intilaized everytime 
        #while True:
        #print('entered here')
        in_mb  = self.tq_diff.get()
        frm_id_list = self.frm_id_q.get()
        batch_tick = time.time()
        np.copyto(self.trt_hin, in_mb.astype(np.float32).ravel())
        comp_tick  = time.time()
        pred = np.array(inference(self.trt_context, self.trt_hin, self.trt_hout, \
                             self.trt_din, self.trt_dout, self.trt_stream))
        t_comp  = 1000 * (time.time() - comp_tick)
        t_batch = 1000 * (time.time() - batch_tick)

        logging.info(" Time %.3f ms " % (t_batch))

            #np.save('../batch_out.npy',pred)
            #ctx.pop()
        
        pred = pred.reshape(8, 16384)    
        
        for j in range(0, len(frm_id_list)):  
            self.processed_count=self.processed_count+1
            if(not(self.processed_count%1000)):
                self.msg1 = "Inference @ {0:.0f}Hz | {1} frames remaining".format(1000/(time.time()-self.t0), (-self.processed_count+self.client.recv_frames))     
                self.t0 = time.time()
                print(self.client.msg2+ " | "+ self.msg1+" \r", end="")
            self.client.server.update(self.client.channel_name, self.client.frame_producer(int(frm_id_list[j]), pred[j]))
            #logging.info("Sent frame id ".format(frm_id_list[j]))