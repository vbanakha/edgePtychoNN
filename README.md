# edgePtychoNN

This code does real-time inference using NVIDIA Jetson AGX Xavier Developer kit on the diffraction patterns streamed out from the X-ray detector. The images are streamed out to the Jetson as PVA stream and 
preprocessed for inference using the Jetson. The embedded GPU system will then perform the inference and sends back the inference outputs as PVA stream to the user interface for viewing. 

main-edge.py performs the inference on the full PtychoNN model 


main-edge-sm.py performs the inference on the reduced PtychoNN model

Models contain the pytorch model used for inference. This model takes in 516x516 input diffraction patterns and outputs the phase (128x128) alone. 
