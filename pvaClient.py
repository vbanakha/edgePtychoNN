import time, queue, threading, sys, os
import argparse, logging
from pvaccess import Channel
from pvaccess import PvObject
import pvaccess as pva
import numpy as np 


class pvaClient:
    def __init__(self, tq_diff, rows = 128, cols = 128):
        self.frames_processed = 0
        self.base_seq_id = None
        self.recv_frames = 0
        self.tq_diff = tq_diff
        self.server = pva.PvaServer()
        self.rows = rows
        self.cols = cols
        self.channel_name = 'pvapy:image1'
        self.server.addRecord(self.channel_name, pva.NtNdArray())
        self.msg1 =''
        self.msg2 =''
        self.t1 = 0
        self.frame_loss =0

    def start(self, pv):
        thread1 = threading.Thread(target=self.monitor(pv), daemon=True)
        thread1.start()
        return thread1

    def frame_producer(self, frame_id, trt_outputs1, extraFieldsPvObject=None):
        
        ## this method is useful to generate a pva stream for the inference outputs. 
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


    def monitor(self, pv):
        uid = pv['uniqueId']

        # ignore the 1st empty frame when use sv simulator
        if self.recv_frames is None:
            self.recv_frames = 0
            return 

        if self.base_seq_id is None: self.base_seq_id = uid
        self.recv_frames += 1

        frm_id= pv['uniqueId']
        dims  = pv['dimension']
        rows  = dims[0]['size']
        cols  = dims[1]['size']
        frame = pv['value'][0]['shortValue']
        
        if(not(self.recv_frames%1000)):
            tmp_frame_loss = self.recv_frames -(uid - self.base_seq_id + 1)
            self.msg2 = " Detector @ {0:.0f}Hz | loss {1:.3f}%".format(1000/(time.time()-self.t1), (-tmp_frame_loss+self.frame_loss)/10)
            self.frame_loss = tmp_frame_loss
            self.t1 = time.time()
            

        self.tq_diff.put((frm_id, frame, rows, cols))


        logging.info("[%.3f] received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     time.time(), uid, self.recv_frames, uid - self.base_seq_id + 1, self.tq_diff.qsize()))