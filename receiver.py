import numpy as np
import adi
import matplotlib.pyplot as plt
from pymongo import MongoClient
import math
import commpy

class Receiver:
    #define power restrictions on signals
    power_threshold = 150 # power in dB the instanteous power must be above to considred a valid input
    max_size = 10

    #declare the raised cosine filter
    # num_taps = 101
    # beta = 0.35
    # Ts = 8
    # t = np.arange(-51,52)
    # h = np.sinc(t/Ts)*np.cos(np.pi*beta*t/Ts)/(1-(2*beta*t/Ts)**2)

    N = int(101)
    alpha = 0.35
    samples_per_symbol = 10
    Fs = 600e3

    h_rrc = commpy.rrcosfilter(N,alpha,samples_per_symbol/Fs,Fs)[1]

    def init_sdr(self):

        try:
            print("Attempting to initialize the SDR...")
            self.sdr = adi.Pluto()
        except:
            print("Unable to initialize SDR, Check connection and drivers")
            return 0
        else:
            return 1

    #connect to the database
    def database_connect(self):
        self.client = MongoClient('mongodb://143.244.155.10:8080')
        self.db = self.client.test
        self.mongo_db_signal_collection = self.db["signal_collection"]
        print("Connected to Database")

    #set the parameters for the sdr to capture signals
    def set_sdr_params(self, center_freq, sampling_freq):
        # self.sdr.gain_control_mode_chan0 = 'slow_attack'
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10
        self.sdr.rx_buffer_size = int(100e3)
        self.sdr.rx_lo = int(center_freq)
        self.sdr.sample_rate = int(sampling_freq)
        self.sdr.rx_rf_bandwidth = int(1e6)

    def set_carrier_freq(self, freq):
        self.sdr.rx_lo = freq

    def set_sampling_rate(self, rate):
        self.sdr.sample_rate = rate

    def set_buffer_size(self, size):
        self.sdr.rx_buffer_size = size

    #display data as a time series
    def disp_time_series(self, data, start_indx, end_indx):
        ax = plt.gca()
        ax.set_ylim([-2e3,2e3])
        t = np.arange(start_indx,end_indx)
        plt.plot(t, data[start_indx:end_indx])
        plt.show()

    #display the signals constellation
    def disp_signal_constellation(self, data, start_indx, end_indx):
        symbols = data[start_indx:end_indx]
        plt.plot(np.real(symbols), np.imag(symbols), '.')
        plt.grid(True)
        plt.show()

    #display the frequency spectrum of a signal
    def disp_freq_spectrum(self, data, start_indx, end_indx):
        X = np.fft.fftshift(np.fft.fft(data[start_indx:end_indx]))
        Xmag = np.abs(X)
        Xphase = np.angle(X)
        f = np.arange((end_indx-start_indx)/-2,(end_indx-start_indx)/2)
        plt.figure(0)
        plt.plot(f,Xmag, '.-')
        # plt.figure(1)
        # plt.plot(f,Xphase, '.-')
        plt.show()

    #store a frame to the database with correct tag
    def insert_data_into_database(self, DB, frame, carrier_signal_frequency, sample_rate, bit_rate, modulation_type):
        document_to_insert = {
            'frame': frame,
            'carrier_signal_freqency': carrier_signal_frequency, 
            'sample_rate': sample_rate,
            'bit_rate': bit_rate,
            'modulation_type': modulation_type
        }

        result = self.mongo_db_client.insert_one(document_to_insert)
        print("Result of insert: %s", result)

    def receive_entire_signal(self):
        data = np.empty((1024))
        estimated_valid_frames = 0
        try:
            while True:
                print('Receiving frames')
                rx_data = self.sdr.rx()
                data = np.append(data,rx_data)
                

        except KeyboardInterrupt:
            print('Done capturing signals')

        print("Shape of Data: " + str(data.shape))
        print("length of Data: " + str(len(data)))
        shaped_data = np.convolve(self.h_rrc,data)
        return shaped_data
    
    def receive_number_samples(self, sample_num):
        data = np.array([])
        
        for i in range(0, int(sample_num/1024)+1, 1):
            rx_data = self.sdr.rx()
            data = np.concatenate((data, rx_data))
        
        requested_data = data[:sample_num]
        
        return requested_data
    
    def extract_signal_data(self, data, frame_size, step_size, num_ramp_steps, num_zero_steps, num_signal_steps, num_signal_samples, p_threshold):
        
        # Average power of each frame will be stored in this array
        avg_power = np.array([])
        
        # Calculate the average power of each frame
        for i in range(0, len(data)-frame_size, step_size):
            frame_avg_power = np.sum(np.abs(data[i:i+frame_size]))/frame_size
            
            frame_power = np.array([frame_avg_power])
            avg_power = np.concatenate([avg_power, frame_power])
        
        ramp_first_valid_frame = -1    # First frame that definitely belongs to the upward ramp
        
        # Check a series of frames and see if the average power increases over the entire series.
        # If it does, this means that all the frames are part of the upward ramp.
        for i in range(len(avg_power)-num_ramp_steps):
            
            previous_power = -1  # The average power of the previous frame
            valid = True         # True if the all the frames belong to the ramp
            
            # Loop through the frames in the series
            for j in range(num_ramp_steps):
                # If the average power of the current frame is greater than that of the previous
                # frame, then the frames could be part of the ramp
                if(avg_power[i+j] > previous_power):
                    previous_power = avg_power[i+j]
                # Otherwise, the series is not part of the ramp
                else:
                    valid = False
                    break
            
            # If the series of frames is part of the ramp:
            if(valid):
                ramp_first_valid_frame = int(i + num_ramp_steps/2)
                break
        
        
        ramp_last_valid_frame = -1
        
        # Find the last valid frame of the ramp
        for i in range(ramp_first_valid_frame,len(avg_power), 1):
            
            # Loop through the frames in the series
            if((avg_power[i] > avg_power[i+1]) and (avg_power[i] > avg_power[i+2]) and (avg_power[i] > avg_power[i+3])):
                ramp_last_valid_frame = i
                break
        
        
        zero_first_valid_frame = -1 # First frame that definitely belongs to the stream of zeros signal
        
        # Next find where the signal goes to zero after the ramp
        for i in range(ramp_last_valid_frame, len(avg_power)-num_zero_steps, 1):
            valid = True    # True if all the frames are below the power threshold
            
            for j in range(num_zero_steps):
                if(avg_power[i+j] < p_threshold):
                    valid = True
                else:
                    valid = False
                    break
            
            # If the series of frames is part of the 0 signal:
            if(valid):
                zero_first_valid_frame = int(i + num_zero_steps/2)
                break
        
        signal_first_valid_frame = -1 # First frame that belongs to the signal we want to record
        
        # Next find where the signal we want to record starts
        for i in range(zero_first_valid_frame, len(avg_power) - num_signal_steps, 1):
            valid = True    # True if all the frames are above the power threshold
            
            for j in range(num_signal_steps):
                if(avg_power[i+j] >= p_threshold):
                    valid = True
                else:
                    valid = False
                    break
            
            if(valid):
                signal_first_valid_frame = i+1
                break
        
        # Now find the sample number that corresponds to the first valid frame of the signal
        signal_start_sample = step_size*signal_first_valid_frame     # First sample of the signal
        signal_end_sample = signal_start_sample + num_signal_samples # Last sample of the signal
        
        # The valid signal data
        signal_data = data[int(signal_start_sample) : int(signal_end_sample)]
        
        return signal_data, signal_start_sample

    #capture one frame of data
    def receive_frame(self,step_count):
        print("Receiving a frame")
        frame_data = np.empty([self.samples_per_frame, 2])
        indx = 0
        data = np.empty((1024))
        while True:
            if self.test:
                rx_data = self.sdr.rx()
            else:
                rx_data = self.test_signal[0 + (1024*step_count):1024 + (1024*step_count)]

            print(rx_data)
            print(rx_data.shape)
            self.input_buffer.store(rx_data)
            print("Here")
            np.append(data,rx_data)

            if(indx > 9):
                
                valid, valid_data = self.process_data()
                if(valid):
                    start_indx = 0
                    end_indx = 0
                    for i in range(2*self.max_size-1):
                        if(valid_data[i] == 1):
                            start_indx = 512*i
                            end_indx = 512*i+1024

                            print( "Displaying Block: "+ str(i) + " With start index: " + str(start_indx) + " and end index of: " + str(end_indx))

                            rx_data = self.input_buffer.get()
                            shaped_data = np.convolve(rx_data,self.h_rrc)
                            return data  

                            # inPhase = np.array(np.real(rx_data[start_indx:end_indx]))
                            # quadPhase = np.array(np.imag(rx_data[start_indx:end_indx]))
                            # print(inPhase)
                            # print(inPhase.shape)

                            # for i in range(self.samples_per_frame):
                            #     frame_data[i,0] = inPhase[i]
                            #     frame_data[i,1] = quadPhase[i]

                            # return(frame_data)
                    

            indx = indx+1 #increment the index, keeps track of how many iterations we have done
            #can be though of as a timeout if we never start storing data before we read this many samples happens then there is an error

            if indx > 1e7:
                print("Error exiting read loop")
                break





        