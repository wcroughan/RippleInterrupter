from scipy import signal, stats
import matplotlib.pyplot as plt
import numpy as np
from spikegadgets import trodesnetwork as tn
import sys
import time
import serial

class MyClient(tn.AbstractModuleClient):
	timestamps = []
	recvquit = False

	def recv_quit(self):
		self.recvquit = True

	def recv_event(self, origin, event, msg):
		if origin == "CameraModule" and event == "newZone":
			self.recvquit = True

		#print(f'origin: {origin}, event: {event}, msg: {msg}')
		print(origin)
		print(event)
		print(msg)



# Constant declarations
BAUDRATE     = 9600
DEFAULT_PORT = '/dev/ttyS1'
WRITE_DATA   = b'\xfd'
DELIVER_HARDWARE_STIM = False

def initializeSerialDevice(port=DEFAULT_PORT):
	ser = serial.Serial(port, BAUDRATE,
			timeout=0,
			stopbits=serial.STOPBITS_ONE,
			bytesize=serial.EIGHTBITS,
			parity=serial.PARITY_NONE
			)
	return ser


if __name__ == "__main__":
	#make filter
	lfp_freq = 1500.0
	filt_len = 0.01 #10ms
	filt_order = int(filt_len * lfp_freq)
	FILT_BUF_LEN = filt_order 
	filt_low = 150.0
	filt_high = 250.0
	sos = signal.butter(filt_order, (filt_low, filt_high), btype='bandpass', analog=False, output='sos', fs=lfp_freq)
	#z1 is keeps track of filter state, so can pass little buffers of a whole signal and the result will be the same as filtering the whole thing in one go
	z1 = signal.sosfilt_zi(sos)

	#init other stuff
	#TODO change this to just the tetrodes we want to record ripple power from.
	trodes = [str(i+1) for i in range(40)]
	#TODO change this to just tetrodes with hippocampal units on them
	spk_trodes = [i + ",0" for i in trodes]
	z1 = np.tile(np.reshape(z1, (filt_order, 1, 2)), (1, len(trodes), 1))
	
	bigbuf = np.zeros((len(trodes), FILT_BUF_LEN))
	bb_idx = 0
	
	timestamp = 0
	zcnt = 0
	REFRAC_PERIOD = 400 #ms
	refrac_end = 0
	SPK_BIN_WIDTH = 5 #ms
	TRODES_SAMP_FREQ = 30000
	
	NUM_SPK_BINS = int(np.round(REFRAC_PERIOD / SPK_BIN_WIDTH))
	bin_time = np.zeros((1,NUM_SPK_BINS))
	spk_cnts = np.zeros((1,NUM_SPK_BINS))
	latest_bin = 0

	#for keeping track of moving average and std of ripple power for each tetrode
	m_n = 1
	m_oldM = np.zeros((1, len(trodes))) 
	m_oldS = np.zeros((1, len(trodes)))

	graph_bool = False
	
	if DELIVER_HARDWARE_STIM:
		serial_port = initializeSerialDevice()

	#connect to trodes
	client = MyClient("RippleInterrupter", "tcp://127.0.0.1", 49152)
	if client.initialize() != 0:
		print("couldn't connect")
		del client
		exit()

	#subscribe to zone events. If user creates a zone, that acts as input to stop this script from running
	client.subscribeToEvent("CameraModule", "newZone")

	#start streaming data
	lfpstream = client.subscribeLFPData(1024, trodes)
	lfpstream.initialize()

	spkstream = client.subscribeSpikesData(1024, trodes)
	spkstream.initialize()

	#after querying for data, trodes fills these with info that was requested
	buf = lfpstream.create_numpy_array()
	spkbuf = lfpstream.create_numpy_array()
	
	print("starting")
	
	allzs = []

	iter = 0
	max_iter = 1000000
	#havent' done any calibrating on this threshold zscore
	#thresh_zrms = 3 * len(trodes)
	thresh_zrms = 1
	
	#subplot 1: spike rate centered on stimulation time
	#subplot 2: ripple-filtered signal every once in a while
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot((REFRAC_PERIOD/2, REFRAC_PERIOD/2), (-10, 10))
	plt.axes((0, -10, REFRAC_PERIOD, 10))
	plt.ion()
	plt.show()

	#TODO switch to this syntax, it's cleaner:
	#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	#ax1.plot(t, sig)
	
	while (not client.recvquit) and iter < max_iter:
		n = lfpstream.available(0)
		for i in range(n):
			timestamp = lfpstream.getData()
			bigbuf[:,bb_idx] = buf[:]
			
			bb_idx += 1
			if bb_idx == FILT_BUF_LEN:
				bb_idx = 0
				rr, z1 = signal.sosfilt(sos, bigbuf, axis=1, zi=z1)
				rms = np.sqrt(np.mean(np.power(rr, 2), axis=1))

				m_n += 1
				m_newM = m_oldM + (rms - m_oldM) / m_n
				m_newS = m_oldS + np.multiply((rms - m_oldM), (rms - m_newM))
				m_oldM = m_newM
				m_oldS = m_newS
				avg_rms = m_newM
				std_rms = np.sqrt(m_newS / (m_n - 1))
				
				zrms = np.divide(rms - avg_rms, std_rms)
				sum_zrms = np.sum(zrms)

				iter = iter + 1
				if iter % 100 == 0:
					pass

				millis = int(round(time.time() * 1000))

				if sum_zrms > thresh_zrms and millis > refrac_end:
					if DELIVER_HARDWARE_STIM:
						serial_port.write(WRITE_DATA)
					
					print("stimulating")
					refrac_end = millis + REFRAC_PERIOD
					graph_time = millis + REFRAC_PERIOD/2
					graph_bool = True
					
					print(iter)
					print(avg_rms)
					print(rms)
					print(std_rms)

					plt.subplot(2,1,2)
					plt.plot(rr[1,:])
					plt.draw()
					plt.pause(0.001)

					plt.subplot(2,1,1)

		n = spkstream.available(0)
		for i in range(n):
			timestamp = spkstream.getData()

			spktime = timestamp.trodes_timestamp * 1000 / TRODES_SAMP_FREQ 
			bin = spktime % NUM_SPK_BINS

			if bin_time[bin] + SPK_BIN_WIDTH < spktime:
				b = bin
				t = spktime
				latest_bin = bin
				while bin_time[b] + SPK_BIN_WIDTH < t:
					bin_time[b] = SPK_BIN_WIDTH * np.floor(t / SPK_BIN_WIDTH)
					spk_cnts[b] = 0

					b -= 1
					if b == -1:
						b = NUM_SPK_BINS - 1
					t -= SPK_BIN_WIDTH

			spk_cnts[bin] += 1

		
		if graph_bool and int(round(time.time() * 1000)) > graph_time:
			graph_bool = False
			if latest_bin < len(bin_time)-1:
				plt.plot(bin_time[latest_bin+1:] - bin_time[latest_bin+1], spk_cnts[latest_bin+1:])
				plt.plot(bin_time[0:latest_bin+1] - bin_time[latest_bin+1], spk_cnts[0:latest_bin+1])
			else:
				plt.plot(bin_time - bin_time[0], spk_cnts)

			plt.draw()
			plt.pause(0.001)












	#cfunctions = [method_name for method_name in dir(client)
        #if callable(getattr(client, method_name))]
	#print(cfunctions)
	#datastream = client.subscribeLFPData(100, ['1','2', '5', '6', '7', '8', '10'])
	#datastream.initialize() #Initialize the streaming object

	#buf = datastream.create_numpy_array()
	#timestamp = 0
	##Continuously get data until Trodes tells us to quit, which triggers the function that flips the "stillrunning" variable
	#iter = 0
	#max_iter = 100
	#while iter < max_iter:
	#	#Get the number of data points that came in
	#	n = datastream.available(1000) #timeout in ms
	#	iter = iter + 1
	#	#Go through and grab all the data packets, processing one at a time
	#	for i in range(n):
	#		timestamp = datastream.getData() 
	#		print(buf)

	#exit()





	
	#t = np.linspace(0, 0.5, 3000)
	#x = (np.sin(2.0*np.pi*0.75*t*(1-t) + 2.1) +
	#	0.4*np.sin(2.0*np.pi*200.0*t + 1) +
	#	0.18*np.cos(2.0*np.pi*3.85*t))
	##x = np.sin(2.0*np.pi*200.0*t)
	#xn = x + np.random.randn(len(t)) * 0.08

	#y_sos, _ = signal.sosfilt(sos, xn, zi=z1)

	#y_sos_cat = []
	#batch_size = 30
	#print(len(t)/batch_size)
	#for i in range(int(len(t)/batch_size)):
	#	new_sig, z1 = signal.sosfilt(sos, xn[i*batch_size:(i+1)*batch_size], zi=z1)
	#	y_sos_cat.extend(new_sig)


	##plt.plot(x, label='clean in')
	##plt.plot(xn, label='noisy in')
	#print(np.sum(y_sos - y_sos_cat))
	#plt.plot(y_sos, 'k', label='SOS')
	#plt.plot(y_sos_cat, 'k', label='SOS_cat')
	#plt.legend(loc='best')
	#plt.show()
	#plt.waitforbuttonpress()
	
	#exit()