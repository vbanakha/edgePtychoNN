from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading
import time
import math
import os 
from multiprocessing import Process, Queue
from framePreProcess import *
from inferPtychoNN import inferPtychoNNtrt
import sys 

from pvaClient import * 
os.environ['export EPICS_PVA_ADDR_LIST']='164.54.128.194'

def main(bsz, cn):


    c = Channel(cn)
    c.setMonitorMaxQueueLength(-1)
    #time.sleep(1)
    #pv_request = ''
    threads = []

    tq_frame = Queue(maxsize=-1)
    batch_q = Queue(maxsize=-1)
    frm_id_q = Queue(maxsize=-1)
    batch_list =[]
    frm_id_list = []
    # initialize pva, it pushes frames into tq_frame
    client = pvaClient(tq_frame)
    c.subscribe('monitor', client.monitor)
    c.startMonitor('')
    
    infer_engine = inferPtychoNNtrt(client, mbsz=bsz, onnx_mdl = '/home/beams/ABABU/ptychoNN-test/new_models/training4_1.8khz/ptychoNN_8.onnx',tq_diff=batch_q, frm_id_q=frm_id_q)

    

    

    while True:
        try:
            frm_id, in_frame, rows, cols = tq_frame.get()
        except queue.Empty:
            continue
        except:
            logging.error("Something else of the Queue went wrong")
            continue

        in_frame = in_frame.reshape(rows, cols)        
        batch_list.append(in_frame)
        
        frm_id_list.append(frm_id)

        

        while(len(batch_list)>=bsz):
            batch_chunk = (np.array(batch_list[:bsz]).astype(np.float32))
            
            batch_frm_id = np.array((frm_id_list[:bsz]))

            batch_q.put(batch_chunk)
            frm_id_q.put(batch_frm_id)
            batch_list=batch_list[bsz:]
            frm_id_list = frm_id_list[bsz:]
    
            
            infer_engine.batch_infer()
            ## create a thread 

            
            
        ## write another thread for sending the frames back to the beamline computer     

    #ctx.pop()
    #return t_batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PtychoNN for phase retreival at the edge')
    parser.add_argument('-gpus', type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-cn',   type=str, default='pvapy:image', help='pva channel name')  # dp_eiger_xrd4:Pva1
    #parser.add_argument('-qs', type=int, default=10000, help='queue size')
    parser.add_argument('-terminal',  type=int, default=0, help='non-zero to print logs to stdout')

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

    bsz = 8

    #scan_810 = np.load('../scan_810.npy')


    

    
    main(bsz, args.cn) 
    

    