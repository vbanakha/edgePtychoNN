import os
import numpy as np
import multiprocessing as mp
import threading
import queue
import time
import pvapy as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor

class InferPtychoNNImageProcessor(AdImageProcessor):

    def __init__(self, configDict={}):
        AdImageProcessor.__init__(self, configDict)
        self.tq_frame_q = mp.Queue(maxsize=-1)
        self.batch_q = mp.Queue(maxsize=-1)
        self.frm_id_q = mp.Queue(maxsize=-1)
        self.nFramesProcessed = 0
        self.nBatchesProcessed = 0
        self.inferTime = 0

        self.bsz = configDict.get('bsz', 8)
        self.onnx_mdl = configDict.get('onnx_mdl', '/home/beams/ABABU/ptychoNN-test/new_models/training4_1.8khz/ptychoNN_8.onnx')
        self.isDone = False

    def inferWorker(self):
        self.logger.debug('Starting infer worker')
        self.gpu = (self.processorId - 1) % 2
        self.logger.debug(f'Using gpu: {self.gpu}')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        from inferPtychoNN import inferPtychoNNtrt
        self.inferEngine = inferPtychoNNtrt(self, mbsz=self.bsz, onnx_mdl=self.onnx_mdl, tq_diff=self.batch_q, frm_id_q=self.frm_id_q)
        self.logger.debug(f'Created infer engine using mbsz={self.bsz} and onnx_mdl={self.onnx_mdl}')
        bsz = self.bsz
        batch_list =[]
        frm_id_list = []

        waitTime = 1
        while True:
            if self.isDone:
                break
            try:
                frm_id, in_frame, ny, nx = self.tq_frame_q.get(block=True, timeout=waitTime)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.isDone = True
                break
            except EOFError:
                self.isDone = True
                break
            except Exception as ex:
                self.logger.error(f'Unexpected error caught: {ex} {type(ex)}')
                break

            batch_list.append(in_frame)
            frm_id_list.append(frm_id)

            while(len(batch_list)>=bsz and not self.isDone):
                batch_chunk = (np.array(batch_list[:bsz]).astype(np.float32))
                batch_frm_id = np.array((frm_id_list[:bsz]))
                self.batch_q.put(batch_chunk)
                self.frm_id_q.put(batch_frm_id)
                batch_list=batch_list[bsz:]
                frm_id_list = frm_id_list[bsz:]

                t0 = time.time()
                self.inferEngine.batch_infer(nx, ny)
                t1 = time.time()
                self.nBatchesProcessed += 1
                self.nFramesProcessed += bsz
                self.inferTime += t1-t0

        try:
            self.logger.debug(f'Stopping infer engine')
            self.inferEngine.stop()
        except Exception as ex:
            self.logger.warn(f'Error stopping infer engine: {ex}')
        self.logger.debug('Infer worker is done')

    def start(self):
        self.inferThread = threading.Thread(target=self.inferWorker)
        self.inferThread.start()

    def stop(self):
        self.logger.debug('Signaling infer worker to stop')
        self.isDone = True

    def configure(self, kwargs):
        self.logger.debug(f'Configuration update: {kwargs}')

    def process(self, pvObject):
        if self.isDone:
            return
        (frameId,image,nx,ny,nz,colorMode,fieldKey) = self.reshapeNtNdArray(pvObject)
        self.tq_frame_q.put((frameId, image, ny, nx))
        return pvObject

    def resetStats(self):
        self.nFramesProcessed = 0
        self.nBatchesProcessed = 0
        self.inferTime = 0

    # Retrieve statistics for user processor
    def getStats(self):
        inferRate = 0
        frameProcessingRate = 0
        if self.nBatchesProcessed  > 0:
            inferRate = self.nBatchesProcessed /self.inferTime
            frameProcessingRate = self.nFramesProcessed/self.inferTime
        nFramesQueued = self.tq_frame_q.qsize()
        return {
            'nFramesProcessed' : self.nFramesProcessed,
            'nBatchesProcessed' : self.nBatchesProcessed,
            'nFramesQueued' : nFramesQueued,
            'inferTime' : self.inferTime,
            'inferRate' : inferRate,
            'frameProcessingRate' : frameProcessingRate
        }

    # Define PVA types for different stats variables
    def getStatsPvaTypes(self):
        return {
            'nFramesProcessed' : pva.UINT,
            'nBatchesProcessed' : pva.UINT,
            'nFramesQueued' : pva.UINT,
            'inferTime' : pva.DOUBLE,
            'inferRate' : pva.DOUBLE,
            'frameProcessingRate' : pva.DOUBLE
        }

