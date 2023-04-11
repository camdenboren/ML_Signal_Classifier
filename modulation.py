# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:55:35 2022

@author: Gjorgi Vitanov
"""

import numpy as np
from scipy import integrate
from scipy.ndimage import gaussian_filter
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt

class Error(Exception):
    """Base class for other exceptions"""
    pass

class InvalidModulationError(Error):
    """Modulation name is not valid."""
    pass

class Modulation:
    
    modulation_type = "random"
    sample_rate = 20e6
    num_symbols = 5000
    ramp_length = 5000
    zero_length = 2500
    sps = 10
    
    # List of valid modulation types
    modulation_types = ["OOK",
                        "4ASK",
                        "8ASK",
                        "BPSK",
                        "QPSK",
                        "8PSK",
                        "16PSK",
                        "32PSK",
                        "16APSK",
                        "32APSK",
                        "64APSK",
                        "128APSK",
                        "16QAM",
                        "32QAM",
                        "64QAM",
                        "128QAM",
                        "256QAM",
                        "AM-DSB-WC",
                        "AM-DSB-SC",
                        "GMSK",
                        "OQPSK"]
    
    def __init__(self, modulation_type="random", num_symbols=5000, ramp_length=5000, zero_length=2500, sps=10, sample_rate=20e6):
        
        # If a random modulation is desired, choose a random modulation from the list of valid modulations
        if(modulation_type == "random"):
            i = np.random.randint(0,len(self.modulation_types))
            modulation_type = self.modulation_types[i]
            print("Modulation type randomly chosen: ", modulation_type)
        
        # Check if the modulation type name is valid
        validModulation = False
        for i in range(len(self.modulation_types)):
            if(modulation_type == self.modulation_types[i]):
                validModulation = True
                break
        
        # If the modulation name is not valid, raise an error
        if(validModulation == False):
            raise InvalidModulationError
        
        self.modulation_type = modulation_type
        self.num_symbols = num_symbols
        self.ramp_length = ramp_length
        self.zero_length = zero_length
        self.sample_rate = sample_rate
    
    def generate_ramp_up(self):
        if(self.ramp_length > 0):
            data = np.arange(0,1,(1/self.ramp_length)) + 1j*0
        else:
            data = np.array([])
        
        return data
    
    def generate_ramp_dn(self):
        if(self.ramp_length > 0):
            data = np.arange(1,0,(-1/self.ramp_length)) + 1j*0
        else:
            data = np.array([])
        
        return data
    
    def generate_zeros(self):
        if(self.zero_length > 0):
            data = np.zeros(self.zero_length) + 1j*0
        else:
            data = np.array([])
        
        
        return data
    
    def generate_input_signal(self):
        
        signal = np.array([])
        if(self.modulation_type == "OOK" or self.modulation_type == "BPSK" or self.modulation_type == "GMSK"):
            signal = np.random.randint(0,2,self.num_symbols)
        elif(self.modulation_type == "4ASK" or self.modulation_type == "QPSK" or self.modulation_type == "OQPSK"):
            signal = np.random.randint(0,4,self.num_symbols)
        elif(self.modulation_type == "8ASK" or self.modulation_type == "8PSK"):
            signal = np.random.randint(0,8,self.num_symbols)
        elif(self.modulation_type == "16PSK" or self.modulation_type == "16APSK" or self.modulation_type == "16QAM"):
            signal = np.random.randint(0,16,self.num_symbols)
        elif(self.modulation_type == "32PSK" or self.modulation_type == "32APSK" or self.modulation_type == "32QAM"):
            signal = np.random.randint(0,32,self.num_symbols)
        elif(self.modulation_type == "64APSK" or self.modulation_type == "64QAM"):
            signal = np.random.randint(0,64,self.num_symbols)
        elif(self.modulation_type == "128APSK" or self.modulation_type == "128QAM"):
            signal = np.random.randint(0,128,self.num_symbols)
        elif(self.modulation_type == "256QAM"):
            signal = np.random.randint(0,256,self.num_symbols)
        elif(self.modulation_type == "AM-DSB-WC" or self.modulation_type == "AM-DSB-SC"):
            t = np.arange(0, self.num_symbols/self.sample_rate, 1/self.sample_rate)
            signal = np.cos(2*np.pi*10e3*t)
            
        return signal
            
    
    def modulate_OOK(self, input_signal):
        modulated_signal = input_signal + 1j*0
        
        return modulated_signal
    
    def modulate_ASK(self, input_signal, bits):
        factor = 1 / (2**bits)
        
        modulated_signal = (input_signal+1)*factor + 1j*0
        
        return modulated_signal
    
    def modulate_PSK(self, input_signal, bits):
        factor = 2**bits
        
        # For BPSK
        if(bits == 1):
            real_signal = input_signal*2 - 1
            modulated_signal = real_signal + 1j*0
        # For all other PSKs:
        else:
            degrees = input_signal*360/factor + 45  # Convert input signal to degrees
            radians = degrees*np.pi/180.0 # sin() and cos() takes in radians
            modulated_signal = np.cos(radians) + 1j*np.sin(radians)
            
        return modulated_signal
    
    def modulate_QAM(self, input_signal, bits):
        
        modulated_signal = np.zeros(len(input_signal), dtype='complex')
        
        # The constellation for the given modulation scheme
        if(bits == 4):
            QAM_Constellation = np.array([-3 - 3*1j, -1 - 3*1j, 3 - 3*1j, 1 - 3*1j, -3 - 1j, -1 - 1j, 3 - 1j, 1 - 1j, -3 + 3*1j, -1 + 3*1j, 3 + 3*1j, 1 + 3*1j, -3 + 1j, -1 + 1j, 3 + 1j, 1 + 1j])
            QAM_Constellation = QAM_Constellation/3
        elif(bits == 5):
            QAM_Constellation = np.array([-3 - 3*1j, -1 - 3*1j, 3 - 3*1j, 1 - 3*1j, -3 - 1j, -1 - 1j, 3 - 1j, 1 - 1j, -3 + 3*1j, -1 + 3*1j, 3 + 3*1j, 1 + 3*1j, -3 + 1j, -1 + 1j, 3 + 1j, 1 + 1j, -3 - 5*1j, -1 - 5*1j, 3 - 5*1j, 1 - 5*1j, 5 - 3*1j, 5 - 1j, 5 + 3*1j, 5 + 1j, -3 + 5*1j, -1 + 5*1j, 3 + 5*1j, 1 + 5*1j, -5 - 3*1j, -5 - 1j, -5 + 3*1j, -5 + 1j])
            QAM_Constellation = QAM_Constellation/5
        elif(bits == 6):
            QAM_Constellation = np.array([-3 - 3*1j, -1 - 3*1j, 3 - 3*1j, 1 - 3*1j, -3 - 1j, -1 - 1j, 3 - 1j, 1 - 1j, -3 + 3*1j, -1 + 3*1j, 3 + 3*1j, 1 + 3*1j, -3 + 1j, -1 + 1j, 3 + 1j, 1 + 1j, -3 - 5*1j, -1 - 5*1j, 3 - 5*1j, 1 - 5*1j, 5 - 3*1j, 5 - 1j, 5 + 3*1j, 5 + 1j, -3 + 5*1j, -1 + 5*1j, 3 + 5*1j, 1 + 5*1j, -5 - 3*1j, -5 - 1j, -5 + 3*1j, -5 + 1j, -5 - 5*1j, 5 - 5*1j, 5 + 5*1j, -5 + 5*1j, -7 - 7*1j, -5 - 7*1j, -3 - 7*1j, -1 - 7*1j, 1 - 7*1j, 3 - 7*1j, 5 - 7*1j, 7 - 7*1j, 7 - 5*1j, 7 - 3*1j, 7 - 1j, 7 + 1j, 7 + 3*1j, 7 + 5*1j, 7 + 7*1j, 5 + 7*1j, 3 + 7*1j, 1 + 7*1j, -1 + 7*1j, -3 + 7*1j, -5 + 7*1j, -7 + 7*1j, -7 + 5*1j, -7 + 3*1j, -7 + 1j, -7 - 1j, -7 - 3*1j, - 7 - 5*1j, -7 - 7*1j])
            QAM_Constellation = QAM_Constellation/7
        elif(bits == 7):
            QAM_Constellation = np.array([-11-7*1j, -11-5*1j, -11-3*1j, -11-1*1j, -11+1*1j, -11+3*1j, -11+5*1j, -11+7*1j, -9-7*1j, -9-5*1j, -9-3*1j, -9-1*1j, -9+1*1j, -9+3*1j, -9+5*1j, -9+7*1j, -7-11*1j, -7-9*1j, -7-7*1j, -7-5*1j, -7-3*1j, -7-1*1j, -7+1*1j, -7+3*1j, -7+5*1j, -7+7*1j, -7+9*1j, -7+11*1j, -5-11*1j, -5-9*1j, -5-7*1j, -5-5*1j, -5-3*1j, -5-1*1j, -5+1*1j, -5+3*1j, -5+5*1j, -5+7*1j, -5+9*1j, -5+11*1j, -3-11*1j, -3-9*1j, -3-7*1j, -3-5*1j, -3-3*1j, -3-1*1j, -3+1*1j, -3+3*1j, -3+5*1j, -3+7*1j, -3+9*1j, -3+11*1j, -1-11*1j, -1-9*1j, -1-7*1j, -1-5*1j, -1-3*1j, -1-1*1j, -1+1*1j, -1+3*1j, -1+5*1j, -1+7*1j, -1+9*1j, -1+11*1j, 1-11*1j, 1-9*1j, 1-7*1j, 1-5*1j, 1-3*1j, 1-1*1j, 1+1*1j, 1+3*1j, 1+5*1j, 1+7*1j, 1+9*1j, 1+11*1j, 3-11*1j, 3-9*1j, 3-7*1j, 3-5*1j, 3-3*1j, 3-1*1j, 3+1*1j, 3+3*1j, 3+5*1j, 3+7*1j, 3+9*1j, 3+11*1j, 5-11*1j, 5-9*1j, 5-7*1j, 5-5*1j, 5-3*1j, 5-1*1j, 5+1*1j, 5+3*1j, 5+5*1j, 5+7*1j, 5+9*1j, 5+11*1j, 7-11*1j, 7-9*1j, 7-7*1j, 7-5*1j, 7-3*1j, 7-1*1j, 7+1*1j, 7+3*1j, 7+5*1j, 7+7*1j, 7+9*1j, 7+11*1j, 9-7*1j, 9-5*1j, 9-3*1j, 9-1*1j, 9+1*1j, 9+3*1j, 9+5*1j, 9+7*1j, 11-7*1j, 11-5*1j, 11-3*1j, 11-1*1j, 11+1*1j, 11+3*1j, 11+5*1j, 11+7*1j])
            QAM_Constellation = QAM_Constellation/11
        elif(bits == 8):
            QAM_Constellation = np.array([-15-15*1j, -15-13*1j, -15-11*1j, -15-9*1j, -15-7*1j, -15-5*1j, -15-3*1j, -15-1*1j, -15+1*1j, -15+3*1j, -15+5*1j, -15+7*1j, -15+9*1j, -15+11*1j, -15+13*1j, -15+15*1j, -13-15*1j, -13-13*1j, -13-11*1j, -13-9*1j, -13-7*1j, -13-5*1j, -13-3*1j, -13-1*1j, -13+1*1j, -13+3*1j, -13+5*1j, -13+7*1j, -13+9*1j, -13+11*1j, -13+13*1j, -13+15*1j, -11-15*1j, -11-13*1j, -11-11*1j, -11-9*1j, -11-7*1j, -11-5*1j, -11-3*1j, -11-1*1j, -11+1*1j, -11+3*1j, -11+5*1j, -11+7*1j, -11+9*1j, -11+11*1j, -11+13*1j, -11+15*1j, -9-15*1j, -9-13*1j, -9-11*1j, -9-9*1j, -9-7*1j, -9-5*1j, -9-3*1j, -9-1*1j, -9+1*1j, -9+3*1j, -9+5*1j, -9+7*1j, -9+9*1j, -9+11*1j, -9+13*1j, -9+15*1j, -7-15*1j, -7-13*1j, -7-11*1j, -7-9*1j, -7-7*1j, -7-5*1j, -7-3*1j, -7-1*1j, -7+1*1j, -7+3*1j, -7+5*1j, -7+7*1j, -7+9*1j, -7+11*1j, -7+13*1j, -7+15*1j, -5-15*1j, -5-13*1j, -5-11*1j, -5-9*1j, -5-7*1j, -5-5*1j, -5-3*1j, -5-1*1j, -5+1*1j, -5+3*1j, -5+5*1j, -5+7*1j, -5+9*1j, -5+11*1j, -5+13*1j, -5+15*1j, -3-15*1j, -3-13*1j, -3-11*1j, -3-9*1j, -3-7*1j, -3-5*1j, -3-3*1j, -3-1*1j, -3+1*1j, -3+3*1j, -3+5*1j, -3+7*1j, -3+9*1j, -3+11*1j, -3+13*1j, -3+15*1j, -1-15*1j, -1-13*1j, -1-11*1j, -1-9*1j, -1-7*1j, -1-5*1j, -1-3*1j, -1-1*1j, -1+1*1j, -1+3*1j, -1+5*1j, -1+7*1j, -1+9*1j, -1+11*1j, -1+13*1j, -1+15*1j, 1-15*1j, 1-13*1j, 1-11*1j, 1-9*1j, 1-7*1j, 1-5*1j, 1-3*1j, 1-1*1j, 1+1*1j, 1+3*1j, 1+5*1j, 1+7*1j, 1+9*1j, 1+11*1j, 1+13*1j, 1+15*1j, 3-15*1j, 3-13*1j, 3-11*1j, 3-9*1j, 3-7*1j, 3-5*1j, 3-3*1j, 3-1*1j, 3+1*1j, 3+3*1j, 3+5*1j, 3+7*1j, 3+9*1j, 3+11*1j, 3+13*1j, 3+15*1j, 5-15*1j, 5-13*1j, 5-11*1j, 5-9*1j, 5-7*1j, 5-5*1j, 5-3*1j, 5-1*1j, 5+1*1j, 5+3*1j, 5+5*1j, 5+7*1j, 5+9*1j, 5+11*1j, 5+13*1j, 5+15*1j, 7-15*1j, 7-13*1j, 7-11*1j, 7-9*1j, 7-7*1j, 7-5*1j, 7-3*1j, 7-1*1j, 7+1*1j, 7+3*1j, 7+5*1j, 7+7*1j, 7+9*1j, 7+11*1j, 7+13*1j, 7+15*1j, 9-15*1j, 9-13*1j, 9-11*1j, 9-9*1j, 9-7*1j, 9-5*1j, 9-3*1j, 9-1*1j, 9+1*1j, 9+3*1j, 9+5*1j, 9+7*1j, 9+9*1j, 9+11*1j, 9+13*1j, 9+15*1j, 11-15*1j, 11-13*1j, 11-11*1j, 11-9*1j, 11-7*1j, 11-5*1j, 11-3*1j, 11-1*1j, 11+1*1j, 11+3*1j, 11+5*1j, 11+7*1j, 11+9*1j, 11+11*1j, 11+13*1j, 11+15*1j, 13-15*1j, 13-13*1j, 13-11*1j, 13-9*1j, 13-7*1j, 13-5*1j, 13-3*1j, 13-1*1j, 13+1*1j, 13+3*1j, 13+5*1j, 13+7*1j, 13+9*1j, 13+11*1j, 13+13*1j, 13+15*1j, 15-15*1j, 15-13*1j, 15-11*1j, 15-9*1j, 15-7*1j, 15-5*1j, 15-3*1j, 15-1*1j, 15+1*1j, 15+3*1j, 15+5*1j, 15+7*1j, 15+9*1j, 15+11*1j, 15+13*1j, 15+15*1j])
            QAM_Constellation = QAM_Constellation/15
        
        # Modulate the input signal accordingly
        for i in range(len(input_signal)):
            QAM_Symbol = QAM_Constellation[input_signal[i]]
            modulated_signal[i] = QAM_Symbol
        
        return modulated_signal
    
    def modulate_APSK(self, input_signal, bits):
        
        modulated_signal = np.zeros(len(input_signal), dtype='complex')
        degrees = np.zeros(len(input_signal))
        radians = np.zeros(len(input_signal))
        
        # 16APSK
        if(bits == 4):
            for i in range(len(input_signal)):
                if input_signal[i] < 4:
                    degrees[i] = input_signal[i]*360/4.0 + 45 # inner ring, radius ratio is R2/R1=2.7, code rate is 5/6
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.37*np.cos(radians[i]) + 0.37*1j*np.sin(radians[i]) # change coefficients to adjust radius ratio / code rate
                else:
                    degrees[i] = input_signal[i]*360/12.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = np.cos(radians[i]) + 1j*np.sin(radians[i])
        # 32 APSK
        if(bits == 5):
            for i in range(len(input_signal)):
                if input_signal[i] < 4:
                    degrees[i] = input_signal[i]*360/4.0 + 45 # inner ring, radius ratio is R3/R1=2.7
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.37*np.cos(radians[i]) + 0.37*1j*np.sin(radians[i]) # change coefficients to adjust radius ratio / code rate
                elif input_signal[i] < 16:
                    degrees[i] = input_signal[i]*360/12.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.685*np.cos(radians[i]) + 0.685*1j*np.sin(radians[i])
                else:
                    degrees[i] = input_signal[i]*360/16.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = np.cos(radians[i]) + 1j*np.sin(radians[i])
        # 64 APSK
        if(bits == 6):
            for i in range(len(input_signal)):
                if input_signal[i] < 4:
                    degrees[i] = input_signal[i]*360/4.0 + 45 # inner ring, radius ratio is R3/R1=2.7
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.37*np.cos(radians[i]) + 0.37*1j*np.sin(radians[i]) # change coefficients to adjust radius ratio / code rate
                elif input_signal[i] < 16:
                    degrees[i] = input_signal[i]*360/12.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.58*np.cos(radians[i]) + 0.58*1j*np.sin(radians[i])
                elif input_signal[i] < 36:
                    degrees[i] = input_signal[i]*360/20.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.79*np.cos(radians[i]) + 0.79*1j*np.sin(radians[i])
                else:
                    degrees[i] = input_signal[i]*360/28.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = np.cos(radians[i]) + 1j*np.sin(radians[i])
        # 128 APSK
        if(bits == 7):
            for i in range(len(input_signal)):
                if input_signal[i] < 16:
                    degrees[i] = input_signal[i]*360/16.0 + 45 # inner ring, radius ratio is R3/R1=2.7
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.37*np.cos(radians[i]) + 0.37*1j*np.sin(radians[i]) # change coefficients to adjust radius ratio / code rate
                elif input_signal[i] < 32:
                    degrees[i] = input_signal[i]*360/16.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.496*np.cos(radians[i]) + 0.496*1j*np.sin(radians[i])
                elif input_signal[i] < 48:
                    degrees[i] = input_signal[i]*360/16.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.622*np.cos(radians[i]) + 0.622*1j*np.sin(radians[i])
                elif input_signal[i] < 64:
                    degrees[i] = input_signal[i]*360/16.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.748*np.cos(radians[i]) + 0.748*1j*np.sin(radians[i])
                elif input_signal[i] < 80:
                    degrees[i] = input_signal[i]*360/16.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = 0.874*np.cos(radians[i]) + 0.874*1j*np.sin(radians[i])
                else:
                    degrees[i] = input_signal[i]*360/48.0 + 45 # outer ring
                    radians[i] = degrees[i]*np.pi/180.0 # sin() and cos() takes in radians
                    modulated_signal[i] = np.cos(radians[i]) + np.sin(radians[i])
        
        return modulated_signal
    
    def modulate_GMSK(self, input_signal):
        
        nrz = input_signal*2 - 1    # Convert binary input signal to non-return-to-zero signal
        
        # Extend the sequence such that every symbol is sps samples
        extended_signal = np.array([])
        
        for i in range(len(input_signal)):
            for j in range(self.sps):
                extended_signal = np.concatenate((extended_signal, [nrz[i]]))
        
        # Integrate the extended signal
        x = np.arange(0, len(extended_signal), 1)
        nrz_integrate = integrate.cumtrapz(extended_signal, x, initial=0)
        
        # Run through Gaussian filter
        filtered_signal = gaussian_filter(input=nrz_integrate, sigma=5)
        
        # Extract I and Q data
        signalI = np.cos(filtered_signal)
        signalQ = np.sin(filtered_signal)
        
        modulated_signal = signalI + 1j*signalQ
        
        return modulated_signal
    
    def modulate_OQPSK(self, input_signal):
        
        # Modulate the input signal with QPSK
        QPSK_data = self.modulate_PSK(input_signal, 2)
        modulated_signal = np.array([])
        
        # Generate the OQPSK modulated signal. Due to the nature of OQPSK, we have 2 samples per symbol
        for i in range(len(QPSK_data)):
            
            real_data = QPSK_data[i].real
            
            if(i==0):
                imag_data1 = 1j*0
            else:
                imag_data1 = QPSK_data[i-1].imag
            
            imag_data2 = QPSK_data[i].imag
            
            data1 = real_data + 1j*imag_data1
            data2 = real_data + 1j*imag_data2
            
            modulated_signal = np.concatenate((modulated_signal, [data1, data2]))
        
        return modulated_signal
    
    def modulate_AM_DSB(self, input_signal, modulation_index, include_carrier):
        if(include_carrier == True):
            modulated_signal = modulation_index*input_signal + 1
        else:
            modulated_signal = modulation_index*input_signal
            
        return modulated_signal
        
    
    def generate_data(self):
        data = np.array([])
        
        # Generate the ramp and zeros signals
        rampUp = self.generate_ramp_up()
        rampDn = self.generate_ramp_dn()
        zeros = self.generate_zeros()
        
        input_signal = self.generate_input_signal()
        modulated_signal = np.array([])
        
        # Generate and modulate an input signal
        if(self.modulation_type == "OOK"):
            modulated_signal = self.modulate_OOK(input_signal)
        elif(self.modulation_type == "BPSK"):
            modulated_signal = self.modulate_PSK(input_signal, 1)
        elif(self.modulation_type == "QPSK"):
            modulated_signal = self.modulate_PSK(input_signal, 2)
        elif(self.modulation_type == "8PSK"):
            modulated_signal = self.modulate_PSK(input_signal, 3)
        elif(self.modulation_type == "16PSK"):
            modulated_signal = self.modulate_PSK(input_signal, 4)
        elif(self.modulation_type == "32PSK"):
            modulated_signal = self.modulate_PSK(input_signal, 5)
        elif(self.modulation_type == "4ASK"):
            modulated_signal = self.modulate_ASK(input_signal, 2)
        elif(self.modulation_type == "8ASK"):
            modulated_signal = self.modulate_ASK(input_signal, 3)
        elif(self.modulation_type == "16APSK"):
            modulated_signal = self.modulate_APSK(input_signal, 4)
        elif(self.modulation_type == "32APSK"):
            modulated_signal = self.modulate_APSK(input_signal, 5)
        elif(self.modulation_type == "64APSK"):
            modulated_signal = self.modulate_APSK(input_signal, 6)
        elif(self.modulation_type == "128APSK"):
            modulated_signal = self.modulate_APSK(input_signal, 7)
        elif(self.modulation_type == "16QAM"):
            modulated_signal = self.modulate_QAM(input_signal, 4)
        elif(self.modulation_type == "32QAM"):
            modulated_signal = self.modulate_QAM(input_signal, 5)
        elif(self.modulation_type == "64QAM"):
            modulated_signal = self.modulate_QAM(input_signal, 6)
        elif(self.modulation_type == "128QAM"):
            modulated_signal = self.modulate_QAM(input_signal, 7)
        elif(self.modulation_type == "256QAM"):
            modulated_signal = self.modulate_QAM(input_signal, 8)
        elif(self.modulation_type == "GMSK"):
            modulated_signal = self.modulate_GMSK(input_signal)
        elif(self.modulation_type == "OQPSK"):
            modulated_signal = self.modulate_OQPSK(input_signal)
        elif(self.modulation_type == "AM-DSB-WC"):
            modulated_signal = self.modulate_AM_DSB(input_signal, 1, True)
        elif(self.modulation_type == "AM-DSB-SC"):
            modulated_signal = self.modulate_AM_DSB(input_signal, 1, False)
        
        # Concatenate all of the signals into one
        full_signal = np.concatenate((rampUp, zeros, modulated_signal, zeros, rampDn))
        
        return full_signal
    
    
    def filter_signal(self, input_signal, num_taps=101, beta=0.35, sps=10, sample_rate=600e3, display_filter=False):
        
        # Make each symbol sps samples long
        x = np.zeros(sps*len(input_signal), dtype='complex')
        for i in range(len(input_signal)):
            # pulse = np.zeros(sps, dtype='complex')
            # pulse[0] = input_signal[i]
            x[i*sps] = input_signal[i]
        
        # Make the filter
        t, h = rrcosfilter(num_taps, beta, sps/sample_rate, sample_rate)
        
        # Plot the filter
        if(display_filter == True):
            plt.figure(255)
            plt.plot(t, h, '.g')
            plt.title('Filter')
        
        x_shaped = np.convolve(x, h)    # Filtered signal
        
        return x_shaped
    
    def normalize_signal(self, input_signal):
        
        maximum = np.amax(np.absolute(input_signal))
        normalized_signal = input_signal / maximum
        
        return normalized_signal
        
        