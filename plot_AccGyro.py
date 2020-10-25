'''
AUTHOR : Luis Pedro Dos Santos
         email : dossantos.luis89@gmail.com
         school : Polytech Sorbonne

DESCRIPTION : Plots the data from the accelerometer and gyrometer in real time

USAGE : python plot_AccGyro.py [mac_adress_of_the_device]

README : If the frame rate is not convenient (the plots don't run smoothly),
         try to change the value of the interval parameter in the FuncAnimation function.

'''


#___________________________________________

from __future__ import print_function
from ctypes import c_void_p, cast, POINTER
from mbientlab.metawear import MetaWear, libmetawear, parse_value, cbindings
from time import sleep
from threading import Event
from sys import argv

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from math import *
import seaborn as sns

#___________________________________________


sns.set(style='whitegrid')

plt.close('all')

states = []
accX = np.array([])
accY = np.array([])
accZ = np.array([])
gyroX = np.array([])
gyroY = np.array([])
gyroZ = np.array([])

class State:
    def __init__(self, device):
        self.device = device
        self.callback = cbindings.FnVoid_VoidP_DataP(self.data_handler)
        self.processor = None

    def data_handler(self, ctx, data):
        values = parse_value(data, n_elem = 2)
        #print("acc: (%.4f,%.4f,%.4f), gyro; (%.4f,%.4f,%.4f)" % (values[0].x, values[0].y, values[0].z, values[1].x, values[1].y, values[1].z))
        
        global accX
        global accY
        global accZ
        global gyroX
        global gyroY
        global gyroZ

        accX = np.append(accX, values[0].x)
        accY = np.append(accY, values[0].y)
        accZ = np.append(accZ, values[0].z)
        gyroX = np.append(gyroX, values[1].x)
        gyroY = np.append(gyroY, values[1].y)
        gyroZ = np.append(gyroZ, values[1].z)


    def setup(self):
        libmetawear.mbl_mw_settings_set_connection_parameters(self.device.board, 7.5, 7.5, 0, 6000)
        sleep(1.5)

        e = Event()

        def processor_created(context, pointer):
            self.processor = pointer
            e.set()
        fn_wrapper = cbindings.FnVoid_VoidP_VoidP(processor_created)
        
        acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
        gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)

        signals = (c_void_p * 1)()
        signals[0] = gyro
        libmetawear.mbl_mw_dataprocessor_fuser_create(acc, signals, 1, None, fn_wrapper)
        e.wait()

        libmetawear.mbl_mw_datasignal_subscribe(self.processor, None, self.callback)

    def start(self):
    
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(self.device.board)
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(self.device.board)

        libmetawear.mbl_mw_gyro_bmi160_start(self.device.board)
        libmetawear.mbl_mw_acc_start(self.device.board)

       
for i in range(len(argv) - 1):
    d = MetaWear(argv[i + 1])
    print("\nConnecting to device...")
    d.connect()
    print("Connected to " + d.address)
    states.append(State(d))

for s in states:
    print("\nConfiguring %s" % (s.device.address))
    s.setup()
    print('Device ready')

for s in states:
    s.start()


# animate(i) : plots the data in the 6 arrays every 'X' interval
def animate(i):
    global accX
    global accY
    global accZ
    global gyroX
    global gyroY
    global gyroZ

    #Accelerometer
    ax1.clear()

    ax1.plot(accX, label='X-Axis')
    ax1.plot(accY, label='Y-Axis')
    ax1.plot(accZ, label='Z-Axis')

    ax1.title.set_text('Accelerometer')
    ax1.grid()
    ax1.legend(loc='upper left')
    ax1.set_ylabel('linear acceleration (g)')

    #sliding window
    ax1.set_xlim(left = max(0, len(accX)-400), right = len(accX)+1)

    #Gyrometer
    ax2.clear()

    ax2.plot(gyroX, label='X-Axis')
    ax2.plot(gyroY, label='Y-Axis')
    ax2.plot(gyroZ, label='Z-Axis')

    ax2.title.set_text('Gyrometer')
    ax2.grid()
    ax2.legend(loc='upper left')
    ax2.set_ylabel('angular velocity (degree/s)')

    #sliding window
    ax2.set_xlim(left = max(0, len(gyroX)-400), right = len(gyroX)+1)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ani = FuncAnimation(fig, animate, interval=20) #change the interval if it doesn't run smoothly

#full screen
figManager = plt.get_current_fig_manager() 
figManager.full_screen_toggle()

plt.show()


print("Resetting devices")
events = []
for s in states:
    e = Event()
    events.append(e)

    s.device.on_disconnect = lambda s: e.set() #Register a handler for disconnect events
    libmetawear.mbl_mw_debug_reset(s.device.board) #Issues a soft reset

for e in events:
    e.wait()