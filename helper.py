import pycuda.driver as cuda
import tensorrt as trt
import logging, torch


def engine_build_from_onnx(onnx_mdl):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(TRT_LOGGER)
    config  = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.max_workspace_size = 1 * (1 << 30) # the maximum size that any layer in the network can use

    network = builder.create_network(EXPLICIT_BATCH)
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    success = parser.parse_from_file(onnx_mdl)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        return None

    return builder.build_engine(network, config)

def mem_allocation(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    
    in_sz = trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size
    h_input  = cuda.pagelocked_empty(in_sz, dtype='float32')
    
    out_sz   = trt.volume(engine.get_binding_shape(1)) * engine.max_batch_size
    h_output = cuda.pagelocked_empty(out_sz, dtype='float32')

    # Allocate device memory for inputs and outputs.
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    return h_input, h_output, d_input, d_output, stream

def inference(context, h_input, h_output, d_input, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Run inference.
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Synchronize the stream
    stream.synchronize()

    # Return the host
    return h_output

## can change this later def pth2onnx(pth):

#def pth2onnx(pth, bsz, in_size):

