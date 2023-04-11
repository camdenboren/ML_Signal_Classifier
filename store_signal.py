import modified_receiver as r
import uuid
import datetime as dt
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from pymongo import MongoClient
import bson


def store_signal():  
    print("Selected Store Signal")
    e = 0
    try:
        signal_source = int(input("Select where you woud like to obtain the signal from\n1. Gather Data from SDR\n2. Load Data From File\n"))
    except:
        pass
    else:
        if signal_source == 1:
            e = collect_from_sdr()

        if signal_source == 2:
            e = load_from_file()

    return e

def load_from_file():
    #grab code from Nate
    print("Loading Data from a File")
    print("Enter File Path")

    return 0
    
def collect_from_sdr():
    rx = r.Receiver(True)

    #commented out for now since I don't have SDR to test
    # sdr_connection =  rx.init_sdr()
    # if sdr_connection == 1:
    #     print("Successfuly initialized SDR")
    # elif sdr_connection == 0:
    #     print("returning to main loop")
    #     return 1
    print("Gathering signal from SDR")
    center_freq = 0
    sampling_frequency = 0
    #User input loop for center freq
    while(True):
        try:
            center_freq = float(input("Enter the desired center frequency, in Hz, to collect signals from: "))
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
            sampling_frequency = float(input("Enter the desired sampling frequency, in Hz: "))
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
    while(True):
        try:
            modulation_input = int(input("Please choose a modulation from this list \
                        \n1. OOK\n2. 4ASK\n3. 8ASK\n4. BPSK \
                        \n5. QPSK \n6. 8PSK\n7. 16PSK \
                        \n8. 32PSK \n9. 16APSK \
                        \n10. 32APSK\n11. 64APSK\n12. 128APSK\n13. 16QAM\n14. 32QAM \
                        \n15. 64QAM\n16. 128QAM\n17. 256QAM\n18. AM-SSB-WC\n19. AM-SSB-SC \
                        \n20. AM-DSB-WC\n21. AM-DSB-SC\n22. FM\n23. GMSK\n24. OQPSK\n"))
        except:
            pass
        else:
            if(modulation_input >= 1 or modulation_input <= 24):
                mod_array = get_mod_arr(modulation_input)
                break
            else:
                print("User input was incorrect")
    
    while(True):
        try:
            user_choice = int(input("Please choose a selection\n1. Default values for SDR signal extraction\n2. Enter custom inputs \
                                    \n3. Collect specific number of samples\n"))
        except:
            pass
        else:
            if(user_choice == 1):
                try:
                    num_sample_input = int(input("Enter the legnth of the signal in samples: "))
                    power_thresh_input = float(input("Enter the desired power threshold: "))
                except:
                    pass
                else:
                    if power_thresh_input <=0 or num_sample_input <=0:
                        print("At least one of your inputs is less than or equal to zero")
                    else:
                        num_signal_samples = num_sample_input
                        power_threshold = power_thresh_input
                        frame_size=500
                        step_size = 250
                        num_ramp_steps = 50
                        num_zero_steps = 50
                        num_signal_steps = 50
                        break
            elif(user_choice == 2): 
                try:
                    frame_size_input = int(input("Enter the desired frame size: "))
                    step_size_input = int(input("Enter the desired step size: "))
                    num_ramp_input = int(input("Enter the desired ramp steps: "))
                    num_zero_input = int(input("Enter the desired zero steps: "))
                    num_signal_input = int(input("Enter the desired signal steps: "))
                    num_sample_input = int(input("Enter the legnth of the signal in samples: "))
                    power_thresh_input = float(input("Enter the desired power threshold: "))

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
                    plt.title("Constellation")
                    rx.disp_signal_constellation_real_imag(signal_data.real, signal_data.imag, 0, len(signal_data)-1)
                    plt.show()
                elif type_of_visual == 2:
                    print("Time Domain of real")
                    plt.title("Time Domain of real")
                    rx.disp_time_series(signal_data.real, 0, len(signal_data)-1)
                    plt.show()
                elif type_of_visual == 3:
                    print("Time domain of imaginary")
                    plt.title("Time domaing of imaginary")
                    rx.disp_time_series(signal_data.imag, 0, len(signal_data)-1)
                    plt.show()
                elif type_of_visual == 4:
                    print("Time domain of magnitude")
                    plt.title("Time domain of magnitude")
                    rx.disp_time_series(np.abs(signal_data), 0, len(signal_data)-1)
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
    
    #input loop to determine wheter to store the signal or not
    while(True):
        try:
            store_data = int(input("Would you like to store this data?\n1. Yes\n2. No\n"))
        except:
            pass
        else:
            if store_data == 1:
                print("Storing Data in database")
                upload_to_database_train(signal_data, mod_array, modulation_input)
                #Here we would store the signal in the database 
                break
            elif store_data == 2:
                print("Not storing Data")
                return 3
            else:
                print("Invalid input, enter again")

    return 0

def upload_to_database_train(signalArrayLong: np.array, modulation: list, modulation_int: int): 
    # Call this function just once with the entire signal you want to upload.. 
    # ID should be randomly generated, also store date/time, and modulation
    client = MongoClient('mongodb://143.244.155.10:8080')
    db = client.test 
    collection = db['test_training_signals'] # change this later

    entireSignalComplexOnly = signalArrayLong

    npArrSigReal = np.real(entireSignalComplexOnly)
    npArrSigImag = np.imag(entireSignalComplexOnly)

    signalArrayTwoDimension = list()

    for i in range(0, len(npArrSigImag)):
        signalArrayTwoDimension.append([npArrSigReal[i], npArrSigImag[i]])

    random_uuid = uuid.uuid1()
    uuid_string = str(random_uuid.int)

    now = dt.datetime.now()
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    numFramesNeeded = math.floor(len(signalArrayTwoDimension) / 1024)
    print(len(signalArrayTwoDimension))
    print(numFramesNeeded)
    tempFrameArray = list()
    listOfDocsToInsert = []

    for i in range(0, numFramesNeeded):
        for j in range(i, 1024 + i):
            tempFrameArray.append(signalArrayTwoDimension[j])
        doc_to_insert = {
            'iq-frame': tempFrameArray,
            'modulation': modulation,
            # 'snr': snr, #somewhere,
            'id': uuid_string,
            'date-time': date_time,
            'training-data': 'true',
            'intOf_Modulation': modulation_int
        }
        listOfDocsToInsert.append(doc_to_insert)
        # if(i % 609 == 0):
        #      result = collection.insert_many(listOfDocsToInsert)
        #      listOfDocsToInsert = []
        # print(result.acknowledged)
        # print(len(bson.BSON.encode(doc_to_insert)))
        print(i)
        tempFrameArray = list()
    result = collection.insert_many(listOfDocsToInsert)
    print(result.acknowledged)
    print(date_time)

def upload_to_database_store(signalArrayLong: np.array): # Not really sure what this one is used for 
    client = MongoClient('mongodb://143.244.155.10:8080')
    db = client.test 
    collection = db['test_store_signals']

    entireSignalComplexOnly = signalArrayLong

    npArrSigReal = np.real(entireSignalComplexOnly)
    npArrSigImag = np.imag(entireSignalComplexOnly)

    signalArrayTwoDimension = list()

    for i in range(0, len(npArrSigImag)):
        signalArrayTwoDimension.append([npArrSigReal[i], npArrSigImag[i]])

    random_uuid = uuid.uuid1()
    uuid_string = str(random_uuid.int)

    now = dt.datetime.now()
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    numFramesNeeded = math.floor(len(signalArrayTwoDimension) / 1024)
    tempFrameArray = list()

    for i in range(0, numFramesNeeded):
        for j in range(i, 1024 + i):
            tempFrameArray.append(signalArrayTwoDimension[j])
        doc_to_insert = {
            'iq-frame': tempFrameArray,
            # 'modulation': modulation, #where does this come from
            # 'snr': snr,#somewhere,
            'id': uuid_string,
            'date-time': date_time,
        }
        result = collection.insert_one(doc_to_insert)
        print(result)
    
def get_mod_arr(i): 
    arr = [0] * 24
    arr[i-1] = 1
    return arr

modulation_classes = json.load(open("./classes-fixed.json", 'r')) # list of modulation classes

def get_mod_from_arr(x):
    return modulation_classes[x.index(1)];


    


