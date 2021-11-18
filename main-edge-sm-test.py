import time, queue, threading, sys, os
import torch, argparse, logging
from pvaccess import Channel
from pvaccess import PvObject
import pvaccess as pva
import numpy as np 

import tensorrt as trt 

sys.path.insert(1, '/home/nvidia-agx/Inference/')

import PtychoNN 
from framePreProcess import * 
from tensorrtcode_batch import *

class pvaClient:
    def __init__(self, nth=1):
        self.last_uid = None
        self.n_missed = 0
        self.n_received = None
        
        self.frame_dims = (516, 516)
        self.debug_frame = np.zeros((128,128), dtype=np.int32)
        self.frame_id = None
        self.trt_engine_path = 'auto_PtychoNN_sm.trt'
        self.resolution = (64,64)
        
        self.server = pva.PvaServer()
        self.channel_name = 'pvapy:image1'
        #self.channel_name_infer = 'pvapy:image2'
        self.server.addRecord(self.channel_name, pva.NtNdArray())
        
        self.current_frame_id = 0
        self.frame_map={}
        self.n_generated_frames = 2
        self.rows = 128
        self.cols = 128
        self.rows1 = 128
        self.cols1 = 128
        self.trt_outputs = ()
        self.max_batch_size = 1
        
        self.base_seq_id = None
        self.frames_processed =0
        self.trt_inference_wrapper = TRTInference(self.trt_engine_path,
        trt_engine_datatype=trt.DataType.FLOAT,
        batch_size=self.max_batch_size)

        self.frame_tq = queue.Queue(maxsize=-1)
        self.processed_tq = queue.Queue(maxsize=-1)
        self.frame_id_tq = queue.Queue(maxsize=-1)
        self.thr_exit = 0
        self.recv_frames = None
        self.avg_times = 0

        for _ in range(nth):
            threading.Thread(target=self.frame_process, daemon=True).start()

    def frame_producer(self, frame_id, trt_outputs1, extraFieldsPvObject=None):
        #for frame_id in range(0, self.n_generated_frames):

        if extraFieldsPvObject is None:
            nda = pva.NtNdArray()
        else:
            nda = pva.NtNdArray(extraFieldsPvObject.getStructureDict())

        nda['uniqueId'] = frame_id
        nda['codec'] = pva.PvCodec('pvapyc', pva.PvInt(5))
        dims = [pva.PvDimension(self.rows, 0, self.rows, 1, False), \
                    pva.PvDimension(self.cols, 0, self.cols, 1, False)]
        nda['dimension'] = dims
        nda['compressedSize'] = self.rows*self.cols
        nda['uncompressedSize'] = self.rows*self.cols
        ts = self.get_timestamp()
        nda['timeStamp'] = ts
        nda['dataTimeStamp'] = ts
        nda['descriptor'] = 'PvaPy Simulated Image'
        nda['value'] = {'floatValue': trt_outputs1.flatten()}
        attrs = [pva.NtAttribute('ColorMode', pva.PvInt(0))]
        nda['attribute'] = attrs
        if extraFieldsPvObject is not None:
            nda.set(extraFieldsPvObject)
            #self.frame_map[frame_id] = nda
        return nda

    def get_timestamp(self):
        s = time.time()
        ns = int((s-int(s))*1000000000)
        s = int(s)
        return pva.PvTimeStamp(s,ns)
        
        
    def frame_process(self, ):
        while self.thr_exit == 0:
            try:
                pv = self.frame_tq.get(block=True, timeout=1)
            except queue.Empty:
                continue
                #logging.error("Queue is empty")
            except:
                #logging.error("Something else of the Queue went wrong")
                continue

            frm_id= pv['uniqueId']
            dims  = pv['dimension']
            rows  = dims[0]['size']
            cols  = dims[1]['size']
            frame = pv['value'][0]['shortValue'].reshape((rows, cols))
            self.frame_tq.task_done()
            
            time0 = time.time()
            processed_frame, pr_frm_id = frame_preprocess(frame, frm_id)
            #print(processed_frame.max())
            #print(processed_frame.sum())
            #self.server.update(self.channel_name, self.frame_producer(frm_id, processed_frame))
            #processed_frame = self.debug_frames 
            print("Time for pre-processing ", (time.time()-time0))
            
            #for _pf in processed_frame:
            self.processed_tq.put(processed_frame)
            self.frame_id_tq.put(frm_id)
                
             

            self.frames_processed += 1
            elapsed = (time.time() - time0)
            in_mb=[]
            in_id =[] ## can be used to resent to the ImageJ
            for i in range(self.max_batch_size):
                _f = self.processed_tq.get()
                _id = self.frame_id_tq.get()
                in_mb.append(_f)
                in_id.append(_id)
                self.processed_tq.task_done()
                self.frame_id_tq.task_done()
            in_mb = np.array(in_mb)
            in_id = np.array(in_id)
            
            if (len(in_mb)==self.max_batch_size):
                #print("entered for inference")
                trt_outputs1, times = self.trt_inference_wrapper.infer(in_mb)
                trt_outputs = np.asarray(trt_outputs1[0])      
                print(trt_outputs.shape)
                print("Execution Times ", times)
            #for _ in in_id:
            self.server.update(self.channel_name, self.frame_producer(frm_id, trt_outputs1[0]))
            print("Sent frame id", frm_id)
            self.avg_times+=(time.time()-time0)
            print("Average time ",(time.time()-time0))

    def monitor(self, pv):
        uid = pv['uniqueId']

        # ignore the 1st empty frame when use sv simulator
        if self.recv_frames is None:
            self.recv_frames = 0
            return 

        if self.base_seq_id is None: self.base_seq_id = uid
        self.recv_frames += 1
        self.frame_tq.put(pv.copy())
        logging.info("[%.3f] received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     time.time(), uid, self.recv_frames, uid - self.base_seq_id + 1, self.frame_tq.qsize()))


#def main_monitor(ch, nth, pv_request):
    
    # give threads seconds to exit
    
    #c.stopMonitor()
    #c.unsubscribe('monitor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-gpus', type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-cn',   type=str, default='QMPX3:test', help='pva channel name')
    parser.add_argument('-qs', type=int, default=10000, help='queue size')
    parser.add_argument('-nth',  type=int, default=1, help='number of threads for frame processes')
    parser.add_argument('-terminal',  type=int, default=0, help='non-zero to print logs to stdout')
    #parser.add_argument('-sf', type=int, default=0, help='specifies how many frames to skip')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logging.basicConfig(filename='edgePtyhcoNN.log', level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)-8s %(message)s',)
    if args.terminal != 0:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
   
    c = Channel(args.cn)
    client = pvaClient(args.nth)
    c.setMonitorMaxQueueLength(args.qs)
    time.sleep(1)
    pv_request = ''

    c.monitor(client.monitor, pv_request)
    time.sleep(1)
     
    client.frame_tq.join()
    client.processed_tq.join()
    client.frame_id_tq.join()

    #client.thr_exit = 1
    time.sleep(10000)

    trt_inference_wrapper.destroy()
    c.stopMonitor()

    

