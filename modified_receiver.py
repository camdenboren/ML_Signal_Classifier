from random import sample
import numpy as np
import adi
import matplotlib.pyplot as plt
from pymongo import MongoClient
import math
import commpy


class Circular_Buffer:

    def __init__(self, max_size):
        self.max = max_size
        self.buffer = np.empty([max_size, 1024], dtype=complex)
        
    def store(self, data):
        for i in range(self.max,1,-1):
            self.shift(i-1,i-2)

        self.buffer[0] = data

    def shift(self, indx_store, indx_out):
        self.buffer[indx_store] = self.buffer[indx_out]

    def get(self):
        return self.buffer.flatten()
    


class Receiver:

    freq_start = 300e6
    freq_end = 900e6
    freq_range = freq_end-freq_start
    freq_step_size = 5e6
    #freq_steps = (freq_range)/freq_step_size
    freq_steps = 1 #set it manually


    freq_sampling_start = 20e6
    freq_sampling_end = 60e6
    freq_sampling_range = freq_sampling_end-freq_sampling_start
    freq_sampling_step_size = 0.25e6
    #freq_sampling_steps = freq_sampling_range/freq_sampling_step_size
    freq_sampling_steps = 1 #set it manually
    #calculate the number of test iterations we are doing
    num_iterations = freq_steps*freq_sampling_steps

    #define frame data
    num_frames_per_iter = 8
    samples_per_frame = 1024

    #define power restrictions on signals
    power_threshold = 150 # power in dB the instanteous power must be above to considred a valid input
    required_valid_samples = 200
    avg_power_period = 100 #the time period over which the signal must have an avg power higher than a certain level
    avg_power_threshold = 2000 #requried avg power over a period of time for a signal to be considered valid
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

    def __init__(self, a):
        
        self.test = a
        self.input_buffer = Circular_Buffer(self.max_size)

        if a:
            print("Declaring Receiver")
            self.sdr = adi.Pluto()
            self.set_sdr_params(2e9,600e3)
        else:
            print("Declaring Test Receiver")
            self.generate_sample_data(1e6)

        #self.database_connect()

    #connect to the database
    def database_connect(self):
        self.client = MongoClient('mongodb://143.244.155.10:8080')
        self.db = self.client.test
        self.mongo_db_signal_collection = self.db["signal_collection"]
        print("Connected to Database")

    #set the parameters for the sdr to capture signals
    def set_sdr_params(self, center_freq, sample_rate):
        # self.sdr.gain_control_mode_chan0 = 'slow_attack'
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10
        self.sdr.rx_buffer_size = int(100e3)
        self.sdr.rx_lo = int(center_freq)
        self.sdr.sample_rate = int(sample_rate)
        
        
        self.sdr.rx_rf_bandwidth = int(1e6)

    def set_carrier_freq(self, freq):
        self.sdr.rx_lo = freq

    def set_sampling_rate(self, rate):
        self.sdr.sample_rate = rate

    def set_buffer_size(self, size):
        self.sdr.rx_buffer_size = size
    #load test data
    def load_sample_data(self, filePath):
        self.test_data = np.loadtxt(filePath, skiprows=1)
        self.complex_test_data = np.empty([100000], dtype=complex)
        self.complex_test_data = self.test_data[0:100000, 0] + self.test_data[0:100000, 1]*1j

    #generate some data to test with
    def generate_sample_data(self,N):
        x_int = np.random.randint(0, 4, N) # 0 to 3
        x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
        x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
        sig = np.cos(x_radians) + 1j*np.sin(x_radians)

        n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # AWGN with unity power
        noise_power = 0.01
        phase_noise = np.random.randn(N) * 0.1 # adjust multiplier for "strength" of phase noise
        self.test_signal = (sig + n * np.sqrt(noise_power))*np.exp(1j*phase_noise)

    #display data as a time series
    def disp_time_series(self, data, start_indx, end_indx):
        ax = plt.gca()
        ax.set_ylim([-2e4,2e4])
        t = np.arange(start_indx,end_indx)
        plt.plot(t, data[start_indx:end_indx])
        plt.show()

    #display the signals constellation
    def disp_signal_constellation(self, data, start_indx, end_indx):
        symbols = data[start_indx:end_indx]
        plt.plot(np.real(symbols), np.imag(symbols), '.')
        plt.grid(True)
        plt.show()

    def disp_signal_constellation_real_imag(self, real, imag, start_indx, end_indx):
        symbols_real = real[start_indx:end_indx]
        symbols_imag = imag[start_indx:end_indx]
        plt.plot(symbols_real, symbols_imag, '.')
        plt.grid(True)
        # plt.title("Signal Constellation")
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

    def process_data(self):
        data = self.input_buffer.get()
        print("Printing data")
        print(data)
        frame_power = np.empty([2*self.max_size-1])
        vaild_frames = np.zeros([2*self.max_size-1])
        print("proccessing data")
        index = 0
        print(self.samples_per_frame*self.max_size)

        for i in range(2*self.max_size-1):
            str_indx = 512*i
            end_indx = 512*i+1024
            abs2 = pow(np.abs(data[str_indx:end_indx]),2)
            frame_power[i] = np.sum(abs2)/1024
            print("calculating block " + str(i) + " Power " + str(frame_power[i]))

        # for i in range(self.max_size):
        #     print(i)
        #     abs2 = pow(np.abs(data[0 + (self.samples_per_frame*index):self.samples_per_frame + (self.samples_per_frame*index)]),2)
        #     frame_power[index] = np.sum(abs2)/1024
        #     print("calculating block " + str(index) + " Power " + str(frame_power[index]))
            
            
        #     if(self.samples_per_frame*(index/2) + self.samples_per_frame*1.5 < self.samples_per_frame*self.max_size):
        #         abs3 = pow(np.abs(data[(int(self.samples_per_frame*(index/2)) + int(self.samples_per_frame*0.5)):(int(self.samples_per_frame*(index/2))+int(self.samples_per_frame*1.5))]),2)
        #         index = index+1
        #         frame_power[index] = np.sum(abs3)/1024
        #         print("calculating midpoint " + str(index) + " Power " + str(frame_power[index]))
                
                

            # index = index + 1

        valid = False
        for i in range(2*self.max_size - 3):
            if(frame_power[i] > self.power_threshold and frame_power[i+1] > self.power_threshold and frame_power[i+2] > self.power_threshold):
                vaild_frames[i+1] = 1
                valid = True

            return valid, vaild_frames

    def receive_entire_signal(self):
        data = np.empty((1024))
        estimated_valid_frames = 0
        try:
            while True:
                print('Receiving')
                rx_data = self.sdr.rx()
                
                data = np.append(data,rx_data)

                power = np.sum(pow(np.abs(rx_data),2))/1024
                
                if(power > self.power_threshold):
                    estimated_valid_frames = estimated_valid_frames+1

        except KeyboardInterrupt:
            print('Done capturing signals')
            # print("Valid Frames: " + str(estimated_valid_frames))
            #pressing ctrl-c will exit the loop
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





        