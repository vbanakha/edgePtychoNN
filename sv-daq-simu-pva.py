import time, threading,argparse
import numpy as np
import pvaccess as pva


from multiprocessing import Queue
import tensorrt as trt

class daqSimuEPICS:

    def __init__(self, npy, daq_freq, nf, nx, ny, runtime, channel_name, start_delay):
        self.arraySize = None
        self.delta_t = 1.0/daq_freq
        self.runtime = runtime
        
        if npy is None:
            self.frames = np.random.randint(0, 256, size=(nf, nx, ny), dtype=np.int16)
        else:
            input_f = np.load(npy)
            self.frames = input_f
            self.frames = np.array(self.frames, dtype=np.dtype('uint16'))

        self.rows, self.cols = self.frames.shape[-2:]

        self.channel_name = channel_name
        self.n_generated_frames = nf
        self.daq_freq = daq_freq
        self.server = pva.PvaServer()
        self.server.addRecord(self.channel_name, pva.NtNdArray())
        self.frame_map = {}
        self.current_frame_id = 0
        self.n_published_frames = 0
        self.start_time = 0
        self.last_published_time = 0
        self.next_publish_time = 0
        self.start_delay = start_delay

    def frame_producer(self, extraFieldsPvObject=None):
        for frame_id in range(0, self.n_generated_frames):

            if extraFieldsPvObject is None:
                nda = pva.NtNdArray()
                
            else:
                nda = pva.NtNdArray(extraFieldsPvObject.getStructureDict())
                
            nda['uniqueId'] = frame_id
            nda['codec'] = pva.PvCodec('pvapyc', pva.PvInt(14))
            dims = [pva.PvDimension(self.rows, 0, self.rows, 1, False), \
                    pva.PvDimension(self.cols, 0, self.cols, 1, False)]
            nda['dimension'] = dims
            nda['descriptor'] = 'PvaPy Simulated Image'
            nda['value'] = {'shortValue': self.frames[frame_id].flatten()}
            if extraFieldsPvObject is not None:
                nda.set(extraFieldsPvObject)
            self.frame_map[frame_id] = nda

    def frame_publisher(self):
        entry_time = time.time()
        cached_frame_id = self.current_frame_id % self.n_generated_frames
        frame = self.frame_map[cached_frame_id]
        frame['uniqueId'] = self.current_frame_id

        # Make sure we do not go too fast
        now = time.time()
        delay = (self.next_publish_time - now)*0.99
        if delay > 0:
            time.sleep(delay)
        self.server.update(self.channel_name, frame)
        self.last_published_time = time.time()
        self.next_publish_time = self.last_published_time + self.delta_t
        self.n_published_frames += 1

        runtime = 0
        rate_correction = 0
        if self.n_published_frames > 1:
            runtime = self.last_published_time - self.start_time
            rate = runtime/(self.n_published_frames - 1)

            # Attempt to correct rate with a bit of magic
            rate_correction = rate - self.delta_t
            if rate_correction > 0:
                rate_correction *= 10

            print("sent frame id %6d @ %.3f (rate: %.4f fps)" % (self.current_frame_id, self.last_published_time, rate))
        else:
            self.start_time = self.last_published_time
            print("sent frame id %6d @ %.3f" % (self.current_frame_id, self.last_published_time))
        self.current_frame_id += 1
        if runtime > self.runtime:
            print("Server will exit after reaching runtime of %s seconds" % (self.runtime))
        else:
            delay = self.delta_t - (time.time()- entry_time) - rate_correction
            threading.Timer(delay, self.frame_publisher).start()

    def start(self):
        threading.Thread(target=self.frame_producer, daemon=True).start()
        self.server.start()
        threading.Timer(self.start_delay, self.frame_publisher).start()

    def stop(self):
        self.server.stop()
        runtime = self.last_published_time - self.start_time
        rate = runtime/(self.n_published_frames - 1)
        print('\nServer runtime: %.4f seconds' % (runtime))
        print('Published frames: %6d @ %.4f fps' % (self.n_published_frames, rate))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate data streaming from detector using EPICS')
    parser.add_argument('-ifn', type=str,   default=None, help='npy file to be streamed')
    parser.add_argument('-fps', type=float, default=1, help='frames per second')
    parser.add_argument('-nx', type=int, default=516, help='number of pixels in x dimension (does not apply if npy file is given)')
    parser.add_argument('-ny', type=int, default=516, help='number of pixels in y dimension (does not apply if npy file is given)')
    parser.add_argument('-nf', type=int, default=1000, help='number of frames to generate')
    parser.add_argument('-rt', type=float, default=300, help='server runtime in seconds')
    parser.add_argument('-cn', type=str, default='QMPX3:test', help='server channel name')
    parser.add_argument('-sd', type=float, default=10.0, help='server start delay')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    daq = daqSimuEPICS(npy=args.ifn, daq_freq=args.fps, nf=args.nf, nx=args.nx, ny=args.ny, runtime=args.rt, channel_name=args.cn, start_delay=args.sd)

    daq.start()
    runtime = args.rt + 2*args.sd
    time.sleep(runtime)
    daq.stop()
    
