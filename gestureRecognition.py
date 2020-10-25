from __future__ import print_function

'''
AUTHOR : Luis Pedro Dos Santos
         email : dossantos.luis89@gmail.com
         school : Polytech Sorbonne

DESCRIPTION : Hand gesture recognition from a 6-gesture set and graphical display of the prediction

USAGE : python gestureRecognition.py [mac_adress_of_the_device]

README : cf. the report

'''



#___________________________________________
''' Libraries '''



#From MbientLab template
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from threading import Event
import keyboard

import platform
import sys

#Data analysis
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from math import *
from time import sleep

#Classification model
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#Preprocessing
from scipy import signal

#User interface
from tkinter import *



#___________________________________________



plt.close('all')


if sys.version_info[0] == 2:
    range = xrange


class State:
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)

    def data_handler(self, ctx, data):
        #print("%s -> %s" % (self.device.address, parse_value(data)))
        #print("samples : ", s.samples)
        self.samples+= 1
        dataAnalysisAfterCallback(data)



#___________________________________________
''' Global variables '''



states = []
arrayDataX = np.array([])
arrayDataY = np.array([])
arrayDataZ = np.array([])

#the points from the segmentation
arrayStartEndLiveMeasures = np.array([])

#the first time we're on the loop : to print on the terminal that the software is ready
softwareReady = False 

#we need to know the first time that the algorithm does a prediction because there won't be any precendent prediction
firstPrediction = True

#precedentPrediction needs to be a global variable because we save the prediction from the precedent loop
precedentPrediction = 'right'



#___________________________________________
''' Callback function, called each time that data from the sensor is retrieved. 'Live' data is analyzed in this function'''



def dataAnalysisAfterCallback(data) :
    global softwareReady
    global arrayStartEndLiveMeasures
    global clf
    global arrayDataX
    global arrayDataY
    global arrayDataZ
    global precedentPrediction
    global firstPrediction

    #Save the data from the sensor to our array
    arrayDataX = np.append(arrayDataX, [parse_value(data).x])
    arrayDataY = np.append(arrayDataY, [parse_value(data).y])
    arrayDataZ = np.append(arrayDataZ, [parse_value(data).z])
    
    #Segmentation input (only for the segmentation of the live data)
    _threshold = 0.07
    _nbMovingAverage = 40

    #The moving average uses the 'X' points before the sample, so we can't work with the data at the very beginning, if we don't have enough points
    if (len(arrayDataX) > 2*_nbMovingAverage) :

        #First time on the loop : the software is ready
        if (softwareReady == False) :
            print('Software ready to be used')
            softwareReady = True
        
        #Create the dataframe from the measures we want to analyze (we only take the '2 * _nbMovingAverage' last points)
        d = {'x-axis (g)' : arrayDataX[len(arrayDataX)-2*_nbMovingAverage :], 'y-axis (g)' : arrayDataY[len(arrayDataX)-2*_nbMovingAverage :], 'z-axis (g)' : arrayDataZ[len(arrayDataX)-2*_nbMovingAverage :]}
        df = pd.DataFrame(data=d)

        #Lowpass filter
        Ax_live_user_beforeNoiseReduc, Ay_live_user_beforeNoiseReduc, Az_live_user_beforeNoiseReduc = lowPassFilter(df)

        #Highpass filter
        Ax_live_user, Ay_live_user, Az_live_user = highPassFilter(Ax_live_user_beforeNoiseReduc, Ay_live_user_beforeNoiseReduc, Az_live_user_beforeNoiseReduc)

        magnitudeLiveData = magnitude(Ax_live_user, Ay_live_user, Az_live_user)

        #Segmentation using the crosses between the threshold and the moving average (taking the '_nbMovingAverage' measures before our sample)
        intermediate = startEndGesturePrecedentMeasures(magnitudeLiveData, _threshold, _nbMovingAverage)        

        #We only work with the data if the segmentation recognized something
        if (len(intermediate) != 0) :
            
            #If one point of the segmentation has already been detected and saved to the array (so we already detected the beginning of the movement)
            if (len(arrayStartEndLiveMeasures) == 1) :

                #And another crossing in the selected window appeared on the segmentation. Therefore, we need to check if it's the same point we already detected (beginning of the movement) or a new one (end of the movement)
                if (len(intermediate) == 1) :

                    #Only if the point just appeared on the window of the segmentation (if the point is on the last 15 measures of the window)
                    if (2*_nbMovingAverage - intermediate[0] < 15) :
                        
                        #Calculates the moving average, to smooth the signal
                        movAvrg = movingAveragePrecedentMeasures(magnitudeLiveData, _nbMovingAverage)

                        #If the curve is following a downwards trend, the point corresponds to the end of the gesture ; and so is different from the one we already saved (corresponding to the beginning of the gesture)
                        if (movAvrg[len(movAvrg)-1] - movAvrg[_nbMovingAverage + int(_nbMovingAverage/3)] < 0) :
                            arrayStartEndLiveMeasures = np.append(arrayStartEndLiveMeasures, s.samples-20)
                            #The moving average is late compared to the signal since we take the precedent measures
                            #So when the moving average crosses the threshold, the actual point of the signal that is crossing is '_nbMovingAverage / 2' points before that
                
                #If the segmentation detected 2 different points. That means we have the start and end of the movement in the same window.
                #Since we already saved the start of the movement, we now save the point corresponding to the end of it.
                if (len(intermediate) == 2) :
                    arrayStartEndLiveMeasures = np.append(arrayStartEndLiveMeasures, s.samples-20)
                      
            #If any point from the segmentation has been saved before
            if (len(arrayStartEndLiveMeasures) == 0) :

                #If the segmentation just returned one point
                if (len(intermediate) == 1) :
                    arrayStartEndLiveMeasures = np.append(arrayStartEndLiveMeasures, s.samples-20)


            #Classification process:

            #If 2 points and so a gesture have been detected
            if (len(arrayStartEndLiveMeasures) == 2) :
                
                #Create the dataframe of the measures corresponding to the gesture
                d = {'x-axis (g)' : arrayDataX[int(arrayStartEndLiveMeasures[0]) : int(arrayStartEndLiveMeasures[1])], 'y-axis (g)' : arrayDataY[int(arrayStartEndLiveMeasures[0]) : int(arrayStartEndLiveMeasures[1])], 'z-axis (g)' : arrayDataZ[int(arrayStartEndLiveMeasures[0]) : int(arrayStartEndLiveMeasures[1])]}
                df = pd.DataFrame(data=d)

                #HANDLING AN ERROR : If the length of the data is too small (< 9), there's a problem with the lowpass filter ('filtfilt' method)
                #But a movement can't be performed that quickly, with only 9 samples.
                #So the movement detected isn't a real gesture. We just ignore it with 'return 0' and reset the array
                if (len(df['x-axis (g)']) < 10) :
                    arrayStartEndLiveMeasures = np.delete(arrayStartEndLiveMeasures, 0)
                    arrayStartEndLiveMeasures = np.delete(arrayStartEndLiveMeasures, 0)
                    return

                #Lowpass filter
                Ax_live_user_beforeNoiseReduc, Ay_live_user_beforeNoiseReduc, Az_live_user_beforeNoiseReduc = lowPassFilter(df)

                #Highpass filter
                Ax_live_user, Ay_live_user, Az_live_user = highPassFilter(Ax_live_user_beforeNoiseReduc, Ay_live_user_beforeNoiseReduc, Az_live_user_beforeNoiseReduc)

                magnitudeLiveData = magnitude(Ax_live_user, Ay_live_user, Az_live_user)
                
                ''' Plotting the detected gesture :

                plt.plot(magnitudeLiveData, label='magnitudeLiveData')
                #plt.plot(Ax_live_user, label='Ax_live_user')
                plt.plot(movingAveragePrecedentMeasures(magnitudeLiveData, _nbMovingAverage), label='movingAveragePrecedentMeasures(magnitudeLiveData, _nbMovingAverage)')
                plt.axhline(y=_threshold, color='r')
                #plt.plot(b, 'ro', label='valueTwoPeaks')
                plt.plot([arrayStartEndLiveMeasures[0]-arrayStartEndLiveMeasures[0], arrayStartEndLiveMeasures[1]-arrayStartEndLiveMeasures[0]], np.zeros(2), 'ro')

                #figManager = plt.get_current_fig_manager()
                #figManager.full_screen_toggle()
                plt.legend()
                plt.title("Segmentation of the live data set")
                plt.show()
                '''

                #String for the features row
                stringOtherFeatures = ['SMA', 'meanTheta', 'meanPhi', 'meanAlpha', 'stdX', 'stdY', 'stdZ', 'meanX', 'meanY', 'meanZ', 'minMaxX', 'minMaxY', 'minMaxZ', 'RMSx', 'RMSy', 'RMSz', 'minX', 'minY','minZ', 'maxX', 'maxY', 'maxZ', 'zeroCrossingsUpX', 'zeroCrossingsUpY', 'zeroCrossingsUpZ', 'zeroCrossingsDownX', 'zeroCrossingsDownY', 'zeroCrossingsDownZ', 'correlationXY', 'correlationXZ', 'correlationYZ', 'kurtosisX', 'kurtosisY', 'kurtosisZ', 'SkewX', 'SkewY', 'SkewZ']
                
                #Checking for an error : if we have calculated more features than the ones expected
                if (len(arrayFeatures(Ax_live_user, Ay_live_user, Az_live_user, 0, len(Ax_live_user)-1)) != len(stringOtherFeatures)) :
                    arrayStartEndLiveMeasures = np.delete(arrayStartEndLiveMeasures, 0)
                    arrayStartEndLiveMeasures = np.delete(arrayStartEndLiveMeasures, 0)
                    return

                #Create the dataframe of the features
                df = pd.DataFrame([arrayFeatures(Ax_live_user, Ay_live_user, Az_live_user, 0, len(Ax_live_user)-1)], columns=list(stringOtherFeatures))

                #We need to know if that's the first prediction of the algorithm because we need to save the precedent prediction to adapt the GUI
                if (firstPrediction) :
                    clfPredict = classificationModelTest(clf, df)
                    precedentPrediction = clfPredict #save the prediction for the next loop

                    if (clfPredict == 'right') :
                        canvas.itemconfig(rightArrow, fill='green')
                    if (clfPredict == 'front') :
                        canvas.itemconfig(frontArrow, fill='green')
                    if (clfPredict == 'left') :
                        canvas.itemconfig(leftArrow, fill='green')
                    if (clfPredict == 'back') :
                        canvas.itemconfig(backArrow, fill='green')
                    if (clfPredict == 'up') :
                        canvas.itemconfig(textCanvas, text='Up', fill='green')
                    if (clfPredict == 'circle') :
                        canvas.itemconfig(textCanvas, text='Circle', fill='green')

                    firstPrediction = False

                else :
                    if (precedentPrediction == 'right') :
                        canvas.itemconfig(rightArrow, fill='black')
                    if (precedentPrediction == 'front') :
                        canvas.itemconfig(frontArrow, fill='black')
                    if (precedentPrediction == 'left') :
                        canvas.itemconfig(leftArrow, fill='black')
                    if (precedentPrediction == 'back') :
                        canvas.itemconfig(backArrow, fill='black')
                    if (precedentPrediction == 'up') :
                        canvas.itemconfig(textCanvas, text='')
                    if (precedentPrediction == 'circle') :
                        canvas.itemconfig(textCanvas, text='')

                    clfPredict = classificationModelTest(clf, df)
                    precedentPrediction = clfPredict

                    if (clfPredict == 'right') :
                        canvas.itemconfig(rightArrow, fill='green')
                    if (clfPredict == 'front') :
                        canvas.itemconfig(frontArrow, fill='green')
                    if (clfPredict == 'left') :
                        canvas.itemconfig(leftArrow, fill='green')
                    if (clfPredict == 'back') :
                        canvas.itemconfig(backArrow, fill='green')
                    if (clfPredict == 'up') :
                        canvas.itemconfig(textCanvas, text='Up', fill='green')
                    if (clfPredict == 'circle') :
                        canvas.itemconfig(textCanvas, text='Circle', fill='green')
                
                #Reset the array containing the 2 points from the segmentation
                arrayStartEndLiveMeasures = np.delete(arrayStartEndLiveMeasures, 0)
                arrayStartEndLiveMeasures = np.delete(arrayStartEndLiveMeasures, 0)

        #Reset the array
        intermediate.clear()

        #Update the GUI
        root.update_idletasks()


            
#___________________________________________
''' Data set with which the classifier will be trained '''



dataForTraining = pd.read_csv("Final Data set.csv")

''' Plot the initial data

plt.figure(figsize=(9, 7))

plt.plot(dataForTraining['x-axis (g)'])
plt.plot(dataForTraining['y-axis (g)'])
plt.plot(dataForTraining['z-axis (g)'])

plt.title('Data from the accelerometer')
plt.legend(['x-axis', 'y-axis', 'z-axis'])
plt.xlabel('number of the measure')
plt.ylabel('linear acceleration (g)')

plt.show()
'''



#___________________________________________
''' LOWPASS FILTER -> Gravitational and Body acceleration '''



sample_rate = 50  # 50 Hz resolution
#signal_lenght = 50*sample_rate  # 50 seconds

Ax = dataForTraining['x-axis (g)']
Ay = dataForTraining['y-axis (g)']
Az = dataForTraining['z-axis (g)']

def butter_lowpass(cutoff, nyq_freq, order):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)
    return y


cutoff_frequency = 0.5
order = 2

Ax_grav = butter_lowpass_filter(Ax, cutoff_frequency, sample_rate/2, order)
Ay_grav = butter_lowpass_filter(Ay, cutoff_frequency, sample_rate/2, order)
Az_grav = butter_lowpass_filter(Az, cutoff_frequency, sample_rate/2, order)

# Difference acts as a special high-pass from a reversed butterworth filter. 
Ax_user_beforeNoiseReduc = np.array(Ax)-np.array(Ax_grav)
Ay_user_beforeNoiseReduc = np.array(Ay)-np.array(Ay_grav)
Az_user_beforeNoiseReduc = np.array(Az)-np.array(Az_grav)

''' Plot the result of the lowpass filter

#X-axis
plt.figure(figsize=(11, 9))
plt.plot(Ax, color='red', label="Original signal, {} samples".format(signal_lenght))
plt.plot(Ax_grav, color='gray', label="Filtered low-pass with cutoff frequency of {} Hz = influence of gravity".format(cutoff_frequency))
plt.plot(Ax_user_beforeNoiseReduc, color='blue', label="What has been removed = dynamic motion the subject is performing")
plt.title("Signal and its filtering")
plt.xlabel('Time (1/50th sec. per tick)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

#Y-axis
plt.figure(figsize=(11, 9))
plt.plot(Ay, color='red', label="Original signal, {} samples".format(signal_lenght))
plt.plot(Ay_grav, color='gray', label="Filtered low-pass with cutoff frequency of {} Hz = influence of gravity".format(cutoff_frequency))
plt.plot(Ay_user_beforeNoiseReduc, color='blue', label="What has been removed = dynamic motion the subject is performing")
plt.title("Signal and its filtering")
plt.xlabel('Time (1/50th sec. per tick)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

#Z-axis
plt.figure(figsize=(11, 9))
plt.plot(Az, color='red', label="Original signal, {} samples".format(signal_lenght))
plt.plot(Az_grav, color='gray', label="Filtered low-pass with cutoff frequency of {} Hz = influence of gravity".format(cutoff_frequency))
plt.plot(Az_user_beforeNoiseReduc, color='blue', label="What has been removed = dynamic motion the subject is performing")
plt.title("Signal and its filtering")
plt.xlabel('Time (1/50th sec. per tick)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
'''



#___________________________________________
''' HIGHPASS FILTER -> Noise reduction '''



def butter_highpass(cutoff, nyq_freq, order):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='highpass')
    return b, a

def butter_highpass_filter(data, cutoff_freq, nyq_freq, order):
    b, a = butter_highpass(cutoff_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)
    return y


cutoff_frequency = 20
order = 2

Ax_noise = butter_highpass_filter(Ax_user_beforeNoiseReduc, cutoff_frequency, sample_rate/2, order)
Ay_noise = butter_highpass_filter(Ay_user_beforeNoiseReduc, cutoff_frequency, sample_rate/2, order)
Az_noise = butter_highpass_filter(Az_user_beforeNoiseReduc, cutoff_frequency, sample_rate/2, order)

# Difference acts as a special low-pass from a reversed butterworth filter. 
Ax_user = np.array(Ax_user_beforeNoiseReduc)-np.array(Ax_noise)
Ay_user = np.array(Ay_user_beforeNoiseReduc)-np.array(Ay_noise)
Az_user = np.array(Az_user_beforeNoiseReduc)-np.array(Az_noise)

''' Plot the result of the highpass filter

#X-axis
plt.figure(figsize=(11, 9))
plt.plot(Ax_user_beforeNoiseReduc, color='red', label="dynamic motion the subject is performing without noise reduction".format(signal_lenght))
plt.plot(Ax_noise, color='gray', label="Filtered high-pass with cutoff frequency of {} Hz = noise".format(cutoff_frequency))
plt.plot(Ax_user, color='blue', label="dynamic motion the subject is performing")
plt.title("Signal and its filtering")
plt.xlabel('Time (1/50th sec. per tick)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

#Y-axis
plt.figure(figsize=(11, 9))
plt.plot(Ay_user_beforeNoiseReduc, color='red', label="dynamic motion the subject is performing without noise reduction".format(signal_lenght))
plt.plot(Ay_noise, color='gray', label="Filtered high-pass with cutoff frequency of {} Hz = noise".format(cutoff_frequency))
plt.plot(Ay_user, color='blue', label="dynamic motion the subject is performing")
plt.title("Signal and its filtering")
plt.xlabel('Time (1/50th sec. per tick)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

#Z-axis
plt.figure(figsize=(11, 9))
plt.plot(Az_user_beforeNoiseReduc, color='red', label="dynamic motion the subject is performing without noise reduction".format(signal_lenght))
plt.plot(Az_noise, color='gray', label="Filtered high-pass with cutoff frequency of {} Hz = noise".format(cutoff_frequency))
plt.plot(Az_user, color='blue', label="dynamic motion the subject is performing")
plt.title("Signal and its filtering")
plt.xlabel('Time (1/50th sec. per tick)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
'''



#___________________________________________
'''After both filters'''



''' Plot the result of both filters

#plt.figure(figsize=(18,14))
plt.figure(1)
plt.subplot(311)
plt.plot(Ax, label='Row data (x-axis)')
plt.plot(Ax_user, label='Data after pre-processing (x-axis)')
#plt.plot(signal.savgol_filter(Ax_user, 53, polyorder=5))
#plt.title("X-Axis before and after the pre-processing")
#plt.xlabel('Measures')
plt.ylabel('Linear acceleration (g)')
plt.legend(loc='lower right')
#plt.show()

#plt.figure(figsize=(18,14))
plt.figure(1)
plt.subplot(312)
plt.plot(Ay, label='Row data (y-axis)')
plt.plot(Ay_user, label='Data after pre-processing (y-axis)')
#plt.plot(signal.savgol_filter(Ay_user, 53, polyorder=5))
#plt.title("Y-Axis before and after the pre-processing")
#plt.xlabel('Measures')
plt.ylabel('Linear acceleration (g)')
plt.legend(loc='lower right')
#plt.show()

#plt.figure(figsize=(18,14))
plt.figure(1)
plt.subplot(313)
plt.plot(Az, label='Row data (z-axis)')
plt.plot(Az_user, label='Data after pre-processing (z-axis)')
#plt.plot(signal.savgol_filter(Az_user, 53, polyorder=5))
#plt.title("Z-Axis before and after the pre-processing")
#plt.xlabel('Measures')
plt.ylabel('Linear acceleration (g)')
plt.legend(loc='lower right')

#figManager = plt.get_current_fig_manager() 
#figManager.full_screen_toggle()
plt.show()
'''


#___________________________________________
''' SEGMENTATION -> Using the crosses between the Simple Moving Average calculated on the magnitude of the signal and a threshold for the live data
                    And the crosses between 2 Simple Moving Averages for the training set '''



#Magnitude of the signal
def magnitude(signal_x, signal_y, signal_z) :
    magnitude = []
    for i in range (len(signal_x)) :
        magnitude.append(sqrt(signal_x[i]*signal_x[i] + signal_y[i]*signal_y[i] + signal_z[i]*signal_z[i]))
    return(magnitude)


#Moving average calculated from an equal number of samples on either side of a central value (this moving average is 'aligned' with the data)
def movingAverage(magnitudeSignal, n) :
    simpleMovingAverage = []
    for i in range (int(n/2)) :
        simpleMovingAverage.append(0)
    
    for i in range (n, len(magnitudeSignal)) :
        sum = 0
        for j in range (i-n, i) :
            sum += magnitudeSignal[j]
        simpleMovingAverage.append(sum/n)

    return (simpleMovingAverage)


#Moving average calculated from the n precedent measures of our sample (so this moving average is shifted in time compared to the signal)
def movingAveragePrecedentMeasures(magnitudeSignal, n) :
    simpleMovingAverage = []
    for i in range (n) :
        simpleMovingAverage.append(0)
    
    for i in range (n, len(magnitudeSignal)) :
        sum = 0
        for j in range (i-n, i) :
            sum += magnitudeSignal[j]
        simpleMovingAverage.append(sum/n)

    return (simpleMovingAverage)


#Segmentation based on the crossings between 2 'centered' moving averages
def startEndGesture2(magnitudeSignal, nbMovingAverage1, nbMovingAverage2) :
    simpleMovingAverage1 = movingAverage(magnitudeSignal, nbMovingAverage1)
    simpleMovingAverage2 = movingAverage(magnitudeSignal, nbMovingAverage2)
    startEndMeasures = []

    #The loop starts at nbMovingAverage2/2 because the points before that are all 0's
    for i in range (150, len(simpleMovingAverage2)-1) :
        if ( (simpleMovingAverage1[i+1] >= simpleMovingAverage2[i+1] and simpleMovingAverage1[i] <= simpleMovingAverage2[i]) or (simpleMovingAverage1[i+1] <= simpleMovingAverage2[i+1] and simpleMovingAverage1[i] >= simpleMovingAverage2[i]) ) :
            startEndMeasures.append(i)
    
    return(startEndMeasures)


#Segmentation based on the crossings between the 'late' moving average (with the precedent points) and a threshold
def startEndGesturePrecedentMeasures(magnitudeSignal, threshold, nbMovingAverage) :
    simpleMovingAverage = movingAveragePrecedentMeasures(magnitudeSignal, nbMovingAverage)
    startEndMeasures = []
    for i in range (len(simpleMovingAverage)-1) :
        if ( (simpleMovingAverage[i+1] >= threshold and simpleMovingAverage[i] <= threshold) or (simpleMovingAverage[i+1] <= threshold and simpleMovingAverage[i] >= threshold) ) :
            startEndMeasures.append(i)
    return(startEndMeasures)



#___________________________________________
''' Calculation of the features '''



'''FEATURE : SMA'''
#Signal Magnitude Area (SMA) --> movement detection

def sma(signal_x, signal_y, signal_z, n1, n2) : #[n1, n2] is the interval of measures from which we want to extract the sma
    sma = 0
    for i in range (n1, n2) :
        sma += ( abs(signal_x[i]) + abs(signal_y[i]) + abs(signal_z[i]) )
    return (sma)


'''FEATURE : TILT ANGLES'''
#Tilt angle --> postural detection

def meanTiltAngles(signal_x, signal_y, signal_z, n1, n2) : #[n1, n2] represents the interval of measures from which we want to exctract the mean value of the angles
    g = 9.81
    
    theta = 0
    phi = 0
    alpha = 0
    
    for i in range (n1, n2) :
        theta += acos(signal_x[i] / g)
        phi += asin(signal_y[i] / g)
        alpha += atan2(signal_z[i], g)
    
    meanTiltAngles=[theta/abs(n2-n1), phi/abs(n2-n1), alpha/abs(n2-n1)]
    
    return(meanTiltAngles)


'''FEATURE : STANDARD DEVIATION'''

def standardDeviation(signal_x, signal_y, signal_z, n1, n2) :
    window_xAxis = signal_x[n1:n2]
    window_yAxis = signal_y[n1:n2]
    window_zAxis = signal_z[n1:n2]
    
    stdX = np.std(window_xAxis)
    stdY = np.std(window_yAxis)
    stdZ = np.std(window_zAxis)
    
    standardDeviation=[stdX, stdY, stdZ]
    
    return(standardDeviation)


'''FEATURE : MEAN'''

def mean(signal_x, signal_y, signal_z, n1, n2) :
    sum_xAxis = 0
    sum_yAxis = 0
    sum_zAxis = 0
    
    for i in range (n1, n2) :
        sum_xAxis += signal_x[i]
        sum_yAxis += signal_y[i]
        sum_zAxis += signal_z[i]
    
    mean=[sum_xAxis/abs(n2-n1), sum_yAxis/abs(n2-n1), sum_zAxis/abs(n2-n1)]
    
    return(mean)


'''FEATURE : MINMAX'''
#Max - min value

def minMax(signal_x, signal_y, signal_z, n1, n2) :
    window_xAxis = signal_x[n1:n2]
    window_yAxis = signal_y[n1:n2]
    window_zAxis = signal_z[n1:n2]
    
    minX = np.amin(window_xAxis)
    minY = np.amin(window_yAxis)
    minZ = np.amin(window_zAxis)
    
    maxX = np.amax(window_xAxis)
    maxY = np.amax(window_yAxis)
    maxZ = np.amax(window_zAxis)
    
    minMax = [maxX-minX, maxY-minY, maxZ-minZ]
    
    return (minMax)


'''FEATURE : RMS'''

def rms(signal_x, signal_y, signal_z, n1, n2) :
    sum_xAxis_squared = 0
    sum_yAxis_squared = 0
    sum_zAxis_squared = 0
    
    for i in range (n1, n2) :
        sum_xAxis_squared += signal_x[i]*signal_x[i]
        sum_yAxis_squared += signal_y[i]*signal_y[i]
        sum_zAxis_squared += signal_z[i]*signal_z[i]
    
    rms = [sqrt(sum_xAxis_squared/abs(n2-n1)), sqrt(sum_yAxis_squared/abs(n2-n1)), sqrt(sum_zAxis_squared/abs(n2-n1))]

    return(rms)


'''FEATURE : MINIMUM VALUE OF AN ARRAY'''

def minOneAxis(signal1, n1, n2) :
    return (np.amin(signal1[n1:n2]))


'''FEATURE : MAXIMUM VALUE OF AN ARRAY'''

def maxOneAxis(signal1, n1, n2) :
    return (np.amax(signal1[n1:n2]))


'''FEATURE : SIGNAL CROSSES THE ZERO AXIS WITH AN UPWARDS TREND'''

def crossZeroAxisCurveGoingUp(signal1, n1, n2) :
    signal1 = signal1[n1:n2]

    #if the window length is too short, the filter can't be applied
    if (n2 - n1 < 5) :
        numberOfCrossingsUp = 0
        for i in range (len(signal1)-1) :     
            if (signal1[i] < 0 and signal1[i+1] > 0) :
                numberOfCrossingsUp += 1        
        return (numberOfCrossingsUp)

    else :
        distance = 2
        polyorder = 5
        numberOfCrossingsUp = 0

        #window length needs to be odd for the filter
        if (((n2-n1-1) % 2) == 0) :
            window = n2-n1-2
        else :
            window = n2-n1-1
    
        #polyorder needs to be smaller than window
        if (polyorder >= window) :
            polyorder = window-1

        filteredSignal = signal.savgol_filter(signal1, window, polyorder)
        for i in range (len(filteredSignal)-1-distance) : 
            if (filteredSignal[i] < 0 and filteredSignal[i+1] > 0 and filteredSignal[i+1+distance] > 0) :
                numberOfCrossingsUp += 1   

        return (numberOfCrossingsUp)


'''FEATURE : SIGNAL CROSSES THE ZERO AXIS WITH A DOWNWARDS TREND'''

def crossZeroAxisCurveGoingDown(signal1, n1, n2) :
    signal1 = signal1[n1:n2]

    #if the window length is too short, the filter can't be applied
    if (n2 - n1 < 5) :
        numberOfCrossingsDown = 0
        for i in range (len(signal1)-1) :     
            if (signal1[i] > 0 and signal1[i+1] < 0) :
                numberOfCrossingsDown += 1    
        return (numberOfCrossingsDown)

    else :
        distance = 2
        polyorder = 5
        numberOfCrossingsDown = 0

        #window length needs to be odd for the filter
        if (((n2-n1-1) % 2) == 0) :
            window = n2-n1-2
        else :
            window = n2-n1-1
    
        #polyorder needs to be smaller than window
        if (polyorder >= window) :
            polyorder = window-1

        filteredSignal = signal.savgol_filter(signal1, window, polyorder)
        for i in range (len(filteredSignal)-1-distance) : 
            if (filteredSignal[i] < 0 and filteredSignal[i+1] > 0 and filteredSignal[i+1+distance] > 0) :
                numberOfCrossingsDown += 1   

        return (numberOfCrossingsDown)


'''FEATURE : CORRELATION BETWEEN TWO AXIS'''

def correlationFeature(signal1, signal2, n1, n2) :
    signal1 = signal1[n1:n2]
    signal2 = signal2[n1:n2]
    correlationMatrix = np.corrcoef(signal1, signal2)

    return(correlationMatrix[0,1])


'''FEATURE : KURTOSIS'''

from scipy.stats import kurtosis
def kurtosisFeature(signal1, n1, n2) :
    signal1 = signal1[n1:n2]
    return (kurtosis(signal1))


'''FEATURE : SKEWNESS'''

from scipy.stats import skew
def skewness(signal1, n1, n2) :
    signal1 = signal1[n1:n2]
    return (skew(signal1))



#___________________________________________
''' Creating the augmented feature vector calculated on the signal in the interval [n1, n2] '''



def arrayFeatures(signal_x, signal_y, signal_z, n1, n2) :
    features = []

    # FEATURE 1 : Signal Magnitude Area (SMA)
    features.append(sma(signal_x, signal_y, signal_z, n1, n2))
    
    # FEATURE 2 : Mean of tilt angles
    features.extend(meanTiltAngles(signal_x, signal_y, signal_z, n1, n2))

    # FEATURE 3 : Standard deviation
    features.extend(standardDeviation(signal_x, signal_y, signal_z, n1, n2))

    # FEATURE 4 : Mean
    features.extend(mean(signal_x, signal_y, signal_z, n1, n2))

    # FEATURE 5 : MinMax
    features.extend(minMax(signal_x, signal_y, signal_z, n1, n2))

    # FEATURE 6 : RMS
    features.extend(rms(signal_x, signal_y, signal_z, n1, n2))

    # FEATURE 7 : Min of the 3 axis
    features.append(minOneAxis(signal_x, n1, n2))
    features.append(minOneAxis(signal_y, n1, n2))
    features.append(minOneAxis(signal_z, n1, n2))
    
    # FEATURE 8 : Max of the 3 axis
    features.append(maxOneAxis(signal_x, n1, n2))
    features.append(maxOneAxis(signal_y, n1, n2))
    features.append(maxOneAxis(signal_z, n1, n2))
    
    # FEATURE 9 : Number of times that the signal crosses the zero axis with an upwards trend
    features.append(crossZeroAxisCurveGoingUp(signal_x, n1, n2))
    features.append(crossZeroAxisCurveGoingUp(signal_y, n1, n2))
    features.append(crossZeroAxisCurveGoingUp(signal_z, n1, n2))

    # FEATURE 10 : Number of times that the signal crosses the zero axis with a downwards trend
    features.append(crossZeroAxisCurveGoingDown(signal_x, n1, n2))
    features.append(crossZeroAxisCurveGoingDown(signal_y, n1, n2))
    features.append(crossZeroAxisCurveGoingDown(signal_z, n1, n2))

    #FEATURE 11 : Correlation between the axis
    features.append(correlationFeature(signal_x, signal_y, n1, n2))
    features.append(correlationFeature(signal_x, signal_z, n1, n2))
    features.append(correlationFeature(signal_y, signal_z, n1, n2))

    #FEATURE 12 : Kurtosis
    features.append(kurtosisFeature(signal_x, n1, n2))
    features.append(kurtosisFeature(signal_y, n1, n2))
    features.append(kurtosisFeature(signal_z, n1, n2))

    #FEATURE 13 : Skewness
    features.append(skewness(signal_x, n1, n2))
    features.append(skewness(signal_y, n1, n2))
    features.append(skewness(signal_z, n1, n2))

    return(features)



#___________________________________________
''' Create a data frame of the features from the training data : the data with which the classifier will be trained with '''



def createDFTrainingSet(signal_x, signal_y, signal_z, startEndMeasures) :
    #Label of the features
    stringFeaturesTitles = ['SMA', 'meanTheta', 'meanPhi', 'meanAlpha', 'stdX', 'stdY', 'stdZ', 'meanX', 'meanY', 'meanZ', 'minMaxX', 'minMaxY', 'minMaxZ', 'RMSx', 'RMSy', 'RMSz', 'minX', 'minY','minZ', 'maxX', 'maxY', 'maxZ', 'zeroCrossingsUpX', 'zeroCrossingsUpY', 'zeroCrossingsUpZ', 'zeroCrossingsDownX', 'zeroCrossingsDownY', 'zeroCrossingsDownZ', 'correlationXY', 'correlationXZ', 'correlationYZ', 'kurtosisX', 'kurtosisY', 'kurtosisZ', 'SkewX', 'SkewY', 'SkewZ']

    #Data frame needs to be created outside the loop (so we can run the command 'df.append(df2, ...)')
    df = pd.DataFrame([arrayFeatures(signal_x, signal_y, signal_z, startEndMeasures[0], startEndMeasures[1])], columns=list(stringFeaturesTitles))

    #Label the feature vector : the first movement was 'right'
    df['target'] = 'right'        

    for i in range ( 1, int(len(startEndMeasures)/2) ) :

        print('loop {} out of {}'.format(i, int(len(startEndMeasures)/2)))

        #Create a data frame row of the features from the detected gesture
        df2 = pd.DataFrame([arrayFeatures(signal_x, signal_y, signal_z, startEndMeasures[2*i], startEndMeasures[(2*i)+1])], columns=list(stringFeaturesTitles))

        #Label the feature vector
        if (i <= 94) :
            df2['target'] = 'right'
        elif(i > 94 and i <= 197) :
            df2['target'] = 'front'
        elif(i > 197 and i <= 321) :
            df2['target'] = 'left'
        elif(i > 321 and i <= 432) :
            df2['target'] = 'back'
        elif(i > 432 and i <= 545) :
            df2['target'] = 'up'
        elif(i > 545) :
            df2['target'] = 'circle'

        df = df.append(df2, ignore_index=True)

    return(df)



#___________________________________________
''' Training the classification algorithm, with the training set '''



def classificationModelTrain(df) :
    X_train = df.drop(['target'], axis='columns')

    #Normalize the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    y_train = df.target

    #Classifier
    #clf = svm.SVC(gamma=0.001, C=100)
    clf = RandomForestClassifier(n_estimators=100, random_state=0)

    clf.fit(X_train, y_train)

    return(clf)



#___________________________________________
''' Testing the classifier on live data '''



def classificationModelTest(clf, dfTestSet) :
    X_test = dfTestSet

    #print('Precision : ', clf.score(X_test, y_test))
    #print('Realite : ', y_test)

    print('\nPrediction : ', clf.predict(X_test))
    
    return(clf.predict(X_test))



#___________________________________________
''' Pre-processing of the live data retrieved from the sensor: '''



#___________________________________________
'''LOWPASS FILTER -> Gravitational and Body acceleration'''



def butter_lowpass(cutoff, nyq_freq, order):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order)
    #print('data : ', data)
    y = signal.filtfilt(b, a, data)
    return y


def lowPassFilter(df) :
    Ax_live = df['x-axis (g)']
    Ay_live = df['y-axis (g)']
    Az_live = df['z-axis (g)']

    cutoff_frequency = 0.5
    order = 2
    
    Ax_live_grav = butter_lowpass_filter(Ax_live, cutoff_frequency, sample_rate/2, order)
    Ay_live_grav = butter_lowpass_filter(Ay_live, cutoff_frequency, sample_rate/2, order)
    Az_live_grav = butter_lowpass_filter(Az_live, cutoff_frequency, sample_rate/2, order)

    # Difference acts as a special high-pass from a reversed butterworth filter. 
    Ax_live_user_beforeNoiseReduc = np.array(Ax_live)-np.array(Ax_live_grav)
    Ay_live_user_beforeNoiseReduc = np.array(Ay_live)-np.array(Ay_live_grav)
    Az_live_user_beforeNoiseReduc = np.array(Az_live)-np.array(Az_live_grav)

    return(Ax_live_user_beforeNoiseReduc, Ay_live_user_beforeNoiseReduc, Az_live_user_beforeNoiseReduc)



#___________________________________________
'''HIGHPASS FILTER -> Noise reduction'''



def butter_highpass(cutoff, nyq_freq, order):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='highpass')
    return b, a


def butter_highpass_filter(data, cutoff_freq, nyq_freq, order):
    b, a = butter_highpass(cutoff_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)
    return y


def highPassFilter(Ax_live_user_beforeNoiseReduc, Ay_live_user_beforeNoiseReduc, Az_live_user_beforeNoiseReduc) :
    cutoff_frequency = 20
    order = 2

    Ax_live_noise = butter_highpass_filter(Ax_live_user_beforeNoiseReduc, cutoff_frequency, sample_rate/2, order)
    Ay_live_noise = butter_highpass_filter(Ay_live_user_beforeNoiseReduc, cutoff_frequency, sample_rate/2, order)
    Az_live_noise = butter_highpass_filter(Az_live_user_beforeNoiseReduc, cutoff_frequency, sample_rate/2, order)

    # Difference acts as a special low-pass from a reversed butterworth filter. 
    Ax_live_user = np.array(Ax_live_user_beforeNoiseReduc)-np.array(Ax_live_noise)
    Ay_live_user = np.array(Ay_live_user_beforeNoiseReduc)-np.array(Ay_live_noise)
    Az_live_user = np.array(Az_live_user_beforeNoiseReduc)-np.array(Az_live_noise)

    return(Ax_live_user, Ay_live_user, Az_live_user)



#___________________________________________
''' MAIN: '''



#___________________________________________
''' Graphical display of the predictions, using tkinter '''



root = Tk()
root.title('Hand Gesture Recognition')

minWidth = 500
minHeight = 350
root.minsize(minWidth, minHeight)

canvas = Canvas(root, width=minWidth, height=270)
canvas.place(anchor=CENTER)
canvas.pack()

#Creating the interface
leftArrow = canvas.create_line((minWidth/2)-140,145,(minWidth/2)-70,145,fill='black',arrow='first',width=15) #left
rightArrow = canvas.create_line((minWidth/2)+70,145,(minWidth/2)+140,145,fill='black',arrow='last',width=15) #right
frontArrow = canvas.create_line(minWidth/2,30,minWidth/2,100,fill='black',arrow='first',width=15) #front
backArrow = canvas.create_line(minWidth/2,190,minWidth/2,260,fill='black',arrow='last',width=15) #back
rectCanvas = canvas.create_rectangle((minWidth/2)-50,120,(minWidth/2)+50,170,fill='white')
textCanvas = canvas.create_text(minWidth/2, 145, anchor=CENTER, font="Purisa", text='')


#Button to close the window
def callback() :
    root.destroy()


button2 = Button(root, text='Close', anchor=CENTER, command=callback)
button2.pack(ipadx=8, ipady=6, pady=30)



#___________________________________________
''' Connecting to the device, using the MAC adress passed as an argument '''



for i in range(len(sys.argv) - 1): #for each device
    d = MetaWear(sys.argv[i + 1])
    print("\nConnecting to device...")
    d.connect()
    print("Connected to " + d.address)
    states.append(State(d))



#___________________________________________
''' Training set : segmentation, features, and training the classifier '''



#Input for the segmentation process
nbMovingAverage = 40
nbMovingAverage2 = 300

#Segmentation:
startEndMeasuresTrainingSet = startEndGesture2(magnitude(Ax_user, Ay_user, Az_user), nbMovingAverage, nbMovingAverage2)

#First point didn't correspond to a movement from the subjects
del startEndMeasuresTrainingSet[0]

#Segments exhibiting a duration shorter than 35 points = 0.7 seconds are filtered out
#(We only want to delete the segments corresponding to gestures so between the start (pair points) and end (odd points) of a gesture, not between two gestures (end of a movement and start of the next one))
segmentsToDelete = []
for i in range (len(startEndMeasuresTrainingSet) - 1) :
    if (startEndMeasuresTrainingSet[i+1] - startEndMeasuresTrainingSet[i] <= 35 and (i%2) == 0 and ((i+1)%2) != 0) :
        segmentsToDelete.append(i)
        segmentsToDelete.append(i+1)

compteur = 0
for i in segmentsToDelete :
    if ((i%2) == 0) :
        del startEndMeasuresTrainingSet[i - compteur]
        del startEndMeasuresTrainingSet[i - compteur]
        compteur += 2


#Features:
dfFeaturesTrainingSet = createDFTrainingSet(Ax_user, Ay_user, Az_user, startEndMeasuresTrainingSet)

#Train the classifier:
clf = classificationModelTrain(dfFeaturesTrainingSet)



#___________________________________________
''' Configuring the device '''



#Starts the measures
for s in states: #for each device
    print("\nConfiguring device...")
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    sleep(1.5)

	#set the parameters for the measure
    libmetawear.mbl_mw_acc_set_odr(s.device.board, 50.0) #sets the output data rate (frequency, in Hz)
    libmetawear.mbl_mw_acc_set_range(s.device.board, 8.0) #sets the full scale range
    libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board) #writes the acceleration settings to the board
    
    dataSignal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board) #retrieves the data signal representing acceleration data
    libmetawear.mbl_mw_datasignal_subscribe(dataSignal, None, s.callback) #subscribes to a data stream, processing messages with the given handler
    
    libmetawear.mbl_mw_acc_enable_acceleration_sampling(s.device.board) #enables acceleration sampling
    libmetawear.mbl_mw_acc_start(s.device.board) #switches the accelerometer to active mode
    print('Device ready')

print('\nConfiguring the software...')


#Turn on the LED during the measures
pattern= LedPattern(repeat_count= Const.LED_REPEAT_INDEFINITELY)
libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)
libmetawear.mbl_mw_led_write_pattern(d.board, byref(pattern), LedColor.RED)
libmetawear.mbl_mw_led_play(d.board)



#___________________________________________
''' infinite loop used to run the application '''

    

root.mainloop()



#___________________________________________
''' Stop the device '''



#Turn of the LED at the end of the measures
libmetawear.mbl_mw_led_stop_and_clear(d.board)

#Stops the measures
for s in states:
    libmetawear.mbl_mw_acc_stop(s.device.board) #switches the accelerometer to standby mode
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board) #disables acceleration sampling

    dataSignal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board) #retrieves the data signal representing acceleration data
    libmetawear.mbl_mw_datasignal_unsubscribe(dataSignal) #unsubscribes from a data stream
    libmetawear.mbl_mw_debug_disconnect(s.device.board) 

#print("\nTotal Samples Received")
#for s in states:
    #print("%s -> %d" % (s.device.address, s.samples))

#Creating dataframe
d = {'x-axis (g)' : arrayDataX, 'y-axis (g)' : arrayDataY, 'z-axis (g)' : arrayDataZ}
df = pd.DataFrame(data=d)
#print(df)

#exit()