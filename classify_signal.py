from statistics import mode
import modified_receiver as r
import uuid
import datetime as dt
import math
import numpy as np
from pymongo import MongoClient
import json
import matplotlib.pyplot as plt
from gc import collect
from ipaddress import collapse_addresses
from pprint import pprint
import tensorflow as tf
import tensorflow_addons as tfa


def classify_signal():
    inputLoop = True
    while(inputLoop):
        user_choice = input("Where would you like to get the signal from?\n1. Get signal from SDR\n2. Get signal from DB\n")
        if(user_choice == "1"):
            model_name = input("Input name for a custom model you want to test if Base models just type 'default': ")
            get_sdr_signal(model_name)
            break
        elif(user_choice == "2"):
            model_name = input("Input name for a custom model you want to test, if Base models just type 'default': ")
            get_signal_from_db(model_name)
            break
        else:
            print("Error, selection could not be found")


def get_sdr_signal(model_name):
    rx = r.Receiver(True)

    #commented out for now since I don't have SDR to test
    # sdr_connection =  rx.init_sdr()
    # if sdr_connection == 1:
    #     print("Successfuly initialized SDR")
    # elif sdr_connection == 0:
    #     print("returning to main loop")
    #     return 1
    # print("Gathering signal from SDR")
    center_freq = 0
    sampling_frequency = 0
    #User input loop for center freq
    while(True):
        try:
            center_freq = float(input("Enter the desired center frequency, in Hz, to collect signals from: \n"))
        except:
            pass
        else:
            if center_freq < 375e6:
                print("The center frequency you selected is too low, enter another")
            elif center_freq > 3.5e9:
                print("The center frequency you selected was too high, enter another")
            else:
                break
    
    #User input loop for center freq
    while(True):
        try:
            sampling_frequency = float(input("Enter the desired sampling frequency, in Hz: \n"))
        except:
            pass
        else:
            if sampling_frequency < 70e3:
                print("The sampling frequency you selected is too low, enter another")
            elif sampling_frequency > 60e6:
                print("The sampling frequency you selected was too high, enter another")
            else:
                break
    
    # Call to get the modulation for signal
    # while(True):
    #     try:
    #         modulation_input = int(input("Please choose a modulation from this list \
    #                     \n1. OOK\n2. 4ASK\n3. 8ASK\n4. BPSK \
    #                     \n5. QPSK \n6. 8PSK\n7. 16PSK \
    #                     \n8. 32PSK \n9. 16APSK \
    #                     \n9. 32APSK\n10. 64APSK\n11. 128APSK\n12. 16QAM\n13. 32QAM \
    #                     \n14. 64QAM\n15. 128QAM\n16. 256QAM\n17. AM-SSB-WC\n18. AM-SSB-SC \
    #                     \n19. AM-DSB-WC\n20. AM-DSB-SC\n21. FM\n22. GMSK\n23. OQPSK\n"))
    #     except:
    #         pass
    #     else:
    #         if(modulation_input >= 1 or modulation_input <= 23):
    #             mod_array = get_mod_arr(modulation_input)
    #             break
    #         else:
    #             print("User input was incorrect")
    
    while(True):
        try:
            user_choice = int(input("Please choose a selection\n1. Default values for SDR signal extraction\n2. Enter custom inputs \
                                    \n3. Collect specific number of samples\n"))
        except:
            pass
        else:
            if(user_choice == 1):
                try:
                    num_sample_input = int(input("Enter the legnth of the signal in samples: \n"))
                    power_thresh_input = float(input("Enter the desired power threshold: \n"))
                except:
                    pass
                else:
                    if power_thresh_input <=0 or num_sample_input <=0:
                        print("At least one of your inputs is less than or equal to zero")
                    else:
                        frame_size=500
                        step_size = 250
                        num_ramp_steps = 50
                        num_zero_steps = 50
                        num_signal_steps = 50
                        num_signal_samples = num_sample_input
                        power_threshold = power_thresh_input
                        break
            elif(user_choice == 2): 
                try:
                    frame_size_input = int(input("Enter the desired frame size: \n"))
                    step_size_input = int(input("Enter the desired step size: \n"))
                    num_ramp_input = int(input("Enter the desired ramp steps: \n"))
                    num_zero_input = int(input("Enter the desired zero steps: \n"))
                    num_signal_input = int(input("Enter the desired signal steps: \n"))
                    num_sample_input = int(input("Enter the legnth of the signal in samples: \n"))
                    power_thresh_input = float(input("Enter the desired power threshold: \n"))

                except:
                    pass
                else:
                    if power_thresh_input > 23170:
                        print("The power threshold is too high")
                    elif frame_size_input <= 0 or step_size_input <= 0 or num_ramp_input <= 0 \
                        or num_zero_input <= 0 or num_signal_input <= 0 or num_sample_input <= 0 or power_thresh_input <= 0:
                        print("At least one of your inputs is less than or equal to zero")
                    else:
                        frame_size= frame_size_input
                        step_size = step_size_input
                        num_ramp_steps = num_ramp_input
                        num_zero_steps = num_zero_input
                        num_signal_steps = num_signal_input
                        num_signal_samples = num_sample_input
                        power_threshold = power_thresh_input
                        break
            elif(user_choice == 3):
                print("do something")
            else:
                break

    
    print("The Center Frequency you selected is: " +str(center_freq)+ " Hz\nThe sampling frequency you selected is: "+str(sampling_frequency)+" Hz")
    #also need a user input to select the modulation type, need help with this

    #update the parameters of the SDR to the ones the user input
    rx.set_sdr_params(center_freq, sampling_frequency)
    rx.sdr.rx_destroy_buffer()

    #might need some more user input to determine how they want to collect signals
    print("Starting Collecting Signals, to stop collecting Signals enter Ctrl-C")
    raw_data = rx.receive_entire_signal()

    print("Finished collecting raw data")
    print("Proccessing Data to find signal")

    #constants don't think these need to be made avaliable to the user
    # frame_size=500
    # step_size = 250
    # num_ramp_steps = 50
    # num_zero_steps = 50
    # num_signal_steps = 50
    # num_signal_samples = 50e3
    # power_threshold = 100 #might need a better way to determine the power threshold of the signal...

    #process the signal and return the output
    signal_data, signal_start_sample = rx.extract_signal_data(raw_data,
                                     frame_size,
                                     step_size,
                                     num_ramp_steps,
                                     num_zero_steps,
                                     num_signal_steps,
                                     num_signal_samples,
                                     power_threshold)

    print("Completed signal proccessing")

    #data visulaization user input
    try:
        data_visual = int(input("Would you like to visualize the data?\n1. Yes\n2. No\n"))
    except:
        pass
    else:
        if data_visual == 1:
            try:
                type_of_visual = int(input("How would you like to visualize the data?\n1. Constellation plot\n2. Time series plot of the real component\n3. Time series plot of the imaginary signal\n4. Time series plot of the signal magnitude\n5. All plots\n"))
            except:
                pass
            else:
                if type_of_visual == 1:
                    print("Constellation")
                    rx.disp_signal_constellation_real_imag(signal_data.real, signal_data.imag, 0, len(signal_data)-1)
                    plt.title("Constellation")
                    plt.show()
                elif type_of_visual == 2:
                    print("Time Domain of real")
                    rx.disp_time_series(signal_data.real, 0, len(signal_data)-1)
                    plt.title("Time Domain of real")
                    plt.show()
                elif type_of_visual == 3:
                    print("Time domaing of imaginary")
                    rx.disp_time_series(signal_data.imag, 0, len(signal_data)-1)
                    plt.title("Time domain of imaginary")
                    plt.show()
                elif type_of_visual == 4:
                    print("Time domain of magnitude")
                    rx.disp_time_series(np.abs(signal_data), 0, len(signal_data)-1)
                    plt.title("Time domain of magnitude")
                    plt.show()
                elif type_of_visual == 5:
                    print("All plots")
                    plt.title("Constellation")
                    rx.disp_signal_constellation_real_imag(signal_data.real, signal_data.imag, 0, len(signal_data)-1)
                    plt.title("Time Domain of real")
                    rx.disp_time_series(signal_data.real, 0, len(signal_data)-1)
                    plt.title("Time domain of imaginary")
                    rx.disp_time_series(signal_data.imag, 0, len(signal_data)-1)
                    plt.title("Time domain of magnitude")
                    rx.disp_time_series(np.abs(signal_data), 0, len(signal_data)-1)
                    plt.show()
                else:
                    print("Invalid input")

        elif data_visual == 2:
            print("Not visualizing data")
        else:
            print("Invalid input not visualizing data")
    while(True):
        user_choice = input("Does the signal visualization look correct?\n1. Yes\n2. No\n")
        if(user_choice == "1"):
            pass_to_identify_signal(signal_data, model_name)
            break
        elif(user_choice == "2"):
            print("Try again?")
            break
        else:
            print("Error. Incorrect user input...")



def get_signal_from_db(model_name):
    while(True):
        try:
            user_mod = int(input("Please choose a modulation from this list \
                        \n1. OOK\n2. 4ASK\n3. 8ASK\n4. BPSK \
                        \n5. QPSK \n6. 8PSK\n7. 16PSK \
                        \n8. 32PSK \n9. 16APSK \
                        \n10. 32APSK\n11. 64APSK\n12. 128APSK\n13. 16QAM\n14. 32QAM \
                        \n15. 64QAM\n16. 128QAM\n17. 256QAM\n18. AM-SSB-WC\n19. AM-SSB-SC \
                        \n20. AM-DSB-WC\n21. AM-DSB-SC\n22. FM\n23. GMSK\n24. OQPSK\n"))
            user_sample = int(input("Please enter the number of sample modulations you would like: \n"))
        except:
            pass
        else:
            if(user_mod >= 1 or user_mod <= 23):
                client = MongoClient('143.244.155.10', 8080)
                db = client.test #this is the database name
                _collection = db['test_collection']
                # mod_array = get_mod_arr(user_mod)

                modulationQueriedList = list(_collection.aggregate([
                    { '$match': { 'intOf_Modulation': user_mod } },
                    { '$sample': { 'size': user_sample } }
                    ]))
                # print(modulationQueriedList)
                # predict_nate_resnet(modulationQueriedList)
                predict_final(modulationQueriedList, model_name)
                break
            else:
                print("User input is incorrect")
    # now get from DB.... send to classify , add model


def get_mod_arr(i): 
    arr = [0] * 24
    arr[i-1] = 1
    return arr

modulation_classes = json.load(open("./classes-fixed.json", 'r')) # list of modulation classes

def get_mod_from_arr(x):
    return modulation_classes[x.index(1)]

def pass_to_identify_signal(signalArrayLong, model_name):
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
        # print(len(tempFrameArray))
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
    # print(np.shape(listOfFrameDocs))
    # predict_nate_resnet(listOfFrameDocs)
    predict_final(listOfFrameDocs, model_name)
    return

# def predict_nate_resnet(my_data):
# # load model
#     # print(my_data)
#     num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    
#     # use resnet if gpu available
#     if (num_gpu >= 1):
#         model = tf.keras.models.load_model('./Complete_Models/resnet_model_mix.h5')
        
#         print('--resnet--')
#         for i in my_data: # predict for each tensor
#             my_data = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
#             # my_data = tf.convert_to_tensor(np.array(i['iq-frame'])) # iq-frame to tensor
#             predictions = model.predict(my_data, batch_size=1, verbose=1) # predict
#             #print(predictions)
#             print("prediction: {}".format(modulation_classes[predictions.argmax()])) # print prediction
#             # print("actual: {}".format(get_mod_from_arr(i['modulation']))) # print actual value    
            
#     else: # use nate's model if no gpu
        
#         model = tf.keras.models.load_model('./nine_layers')
#         print('--nate--')
#         for i in my_data: # predict for each tensor
#             my_data = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
#             # my_data = tf.convert_to_tensor(np.array(i['iq-frame'])) # iq-frame to tensor
#             predictions = model.predict(my_data, batch_size=1, verbose=1) # predict
#             #print(predictions)
#             print("prediction: {}".format(modulation_classes[predictions.argmax()])) # print prediction
#             # print("actual: {}".format(get_mod_from_arr(i['modulation']))) # print actual value 


def predict_final(my_data, customModel = "default"):
    
    num_gpu = len(tf.config.experimental.list_physical_devices('GPU')) # get number of gpu's available    
    
    predictions_high_snr = [] # high snr model predictions
    predictions_low_snr = [] # low snr model predictions
    
    if(customModel == "default"):
    # use resnet if gpu available
        if (num_gpu >= 1):

            try:
                with tf.device('/device:GPU:0'):
                    resnet_model = tf.keras.models.load_model('./Complete_Models/resnet_model_mix.h5')
                    #print('--resnet--')
                    for i in my_data: # predict for each tensor
                        current_sample = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
                        predictions_high_snr.append(resnet_model.predict(current_sample, batch_size=1, verbose=1)) # append high snr predictions
            except RuntimeError as e:
                print(e)

        else: # use nate's model if no gpu
            nate_model = tf.keras.models.load_model('./nine_layers')
            #print('--nate--')
            for i in my_data: # predict for each tensor
                current_sample = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
                predictions_high_snr.append(nate_model.predict(current_sample, batch_size=1, verbose=1)) # append high snr predictions    

        # convert list[dict] structure to tensors
        for i in my_data: # predict for each tensor
            trafo_model = tf.keras.models.load_model('./Complete_Models/trafo_model')
            current_sample = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
            predictions_low_snr.append(trafo_model.predict(current_sample, batch_size=1, verbose=1)) # append low snr predictions


        # loop through samples for final prediction
        for i in range(len(my_data)):
            hi = predictions_high_snr[i].argmax() # high snr model predicted index
            li = predictions_low_snr[i].argmax() # low snr model predicted index

            hi_val = predictions_high_snr[i][0][hi] # high snr model probability
            low_val = predictions_low_snr[i][0][li] # low snr model probability

            # print index, probability, and prediction for each model
            print('high snr index: ' + str(hi) + ' | value: ' + str(hi_val) + ' | prediction: {}'.format(modulation_classes[hi]))
            print('low snr index: ' + str(li) + ' | value: ' + str(low_val)  + ' | prediction: {}'.format(modulation_classes[li]))

            # print whichever model's probability is higher for final prediction
            if (hi_val >= low_val):
                print("final prediction: {}".format(modulation_classes[hi])) # print high model final prediction
            else:
                print("final prediction: {}".format(modulation_classes[li])) # print high model final prediction

            # print("actual: {}\n\n".format(get_mod_from_arr(my_data[i]['modulation']))) # print actual value
    
    else: 
        custom_model = tf.keras.models.load_model(f'./models/{customModel}')
        modulation_classes_model = json.load(open(f"./models/{customModel}/{customModel}.json", 'r')) # list of modulation classes
        #print('--nate--')
        for i in my_data: # predict for each tensor
            current_sample = tf.convert_to_tensor(np.array(i['iq-frame']).reshape(1,1024,2)) # iq-frame to tensor
            predictions_high_snr.append(custom_model.predict(current_sample, batch_size=1, verbose=1)) # append high snr predictions  
        
        for i in range(len(my_data)):
            hi = predictions_high_snr[i].argmax() # high snr model predicted index
            
            hi_val = predictions_high_snr[i][0][hi] # high snr model probability
    
            # print index, probability, and prediction for each model
            print('high snr index: ' + str(hi) + ' | value: ' + str(hi_val) + ' | prediction: {}'.format(modulation_classes_model[hi]))