
import modified_receiver as r
import uuid
import datetime as dt
import json
from gc import collect
from ipaddress import collapse_addresses
from pprint import pprint
import tensorflow as tf
import tensorflow_addons as tfa
import math


import numpy as np

def get_mod_arr(i): 
    arr = [0] * 24
    arr[i-1] = 1
    return arr

modulation_classes = json.load(open("./classes-fixed.json", 'r')) # list of modulation classes

def get_mod_from_arr(x):
    return modulation_classes[x];

def pass_to_identify_signal(signalArrayLong):
    # Call this function when user is calling SDR 
    # Call the function that identifies a signal against model.. in progress

    entireSignalComplexOnly = signalArrayLong

    npArrSigReal = np.real(entireSignalComplexOnly)
    npArrSigImag = np.imag(entireSignalComplexOnly)

    signalArrayTwoDimension = list()

    for i in range(0, len(npArrSigImag)):
        signalArrayTwoDimension.append([npArrSigReal[i], npArrSigImag[i]])

    listOfFrameDocs = list()

    random_uuid = uuid.uuid1()
    uuid_string = str(random_uuid.int)

    now = dt.datetime.now()
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    numFramesNeeded = int(len(signalArrayTwoDimension) / 1024)
    tempFrameArray = list()

    for i in range(0, numFramesNeeded):
        for j in range(i, 1024 + i):
            tempFrameArray.append(signalArrayTwoDimension[j])
        print(len(tempFrameArray))
        doc_to_insert = {
            'iq-frame': tempFrameArray,
            'modulation': ([0] * 24), #Nothing
            'snr': 0, #Nothing
            'id': uuid_string,
            'date-time': date_time,
        }
        listOfFrameDocs.append(doc_to_insert)
        tempFrameArray = list()

    # print(listOfFrameDocs)
    middle_index = int(len(listOfFrameDocs) / 2)
    # print(listOfFrameDocs[middle_index])
    print(np.shape(listOfFrameDocs))
    predict_nate_resnet(listOfFrameDocs)
    return

def predict_nate_resnet(my_data):
# load model
    # print(my_data)
    num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    
    # use resnet if gpu available
    if (num_gpu >= 1):
        model = tf.keras.models.load_model('./Complete_Models/resnet_model_mix.h5')
        
        print('--resnet--')
        for i in my_data: # predict for each tensor
            my_data = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
            # my_data = tf.convert_to_tensor(np.array(i['iq-frame'])) # iq-frame to tensor
            predictions = model.predict(my_data, batch_size=1, verbose=1) # predict
            #print(predictions)
            print("prediction: {}".format(modulation_classes[predictions.argmax()])) # print prediction
            # print("actual: {}".format(get_mod_from_arr(i['modulation']))) # print actual value    
            
    else: # use nate's model if no gpu
        
        model = tf.keras.models.load_model('./nine_layers')
        print('--nate--')
        for i in my_data: # predict for each tensor
            my_data = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
            # my_data = tf.convert_to_tensor(np.array(i['iq-frame'])) # iq-frame to tensor
            predictions = model.predict(my_data, batch_size=1, verbose=1) # predict
            #print(predictions)
            print("prediction: {}".format(modulation_classes[predictions.argmax()])) # print prediction
            # print("actual: {}".format(get_mod_from_arr(i['modulation']))) # print actual value 


carrier_freq = 2e9
sampling_rate = 600e3


rx = r.Receiver(True)

#commented out for now since I don't have SDR to test
# sdr_connection =  rx.init_sdr()
# if sdr_connection == 1:
#     print("Successfuly initialized SDR")
# print("Gathering signal from SDR")
# center_freq = 0
# sampling_frequency = 0

print("Collecting entire signal")
full_data = rx.receive_entire_signal()
# full_data = rx.receive_number_samples(50000)
print("Done collecting entire signal")
print("Extracting signal data")

# %% Process

frame_size=500
step_size = 250
num_ramp_steps = 50
num_zero_steps = 50
num_signal_steps = 50
num_signal_samples = 50e3
power_threshold = 30

signal_start_sample = 0

signal_data, signal_start_sample = rx.extract_signal_data(full_data,
                                     frame_size,
                                     step_size,
                                     num_ramp_steps,
                                     num_zero_steps,
                                     num_signal_steps,
                                     num_signal_samples,
                                     power_threshold)
print("Done extracting singal data")
print("First signal sample:", signal_start_sample)

#rx.disp_signal_constellation(signal_data, 0, len(signal_data)-1)
rx.disp_time_series(full_data, 0, len(full_data)-1)
rx.disp_time_series(signal_data.real, 0, len(signal_data)-1)
rx.disp_time_series(signal_data.imag, 0, len(signal_data)-1)
# rx.disp_time_series(np.abs(signal_data), 0, len(signal_data)-1)


pass_to_identify_signal(signal_data)