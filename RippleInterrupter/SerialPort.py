"""
Module for communicating with serial port for communication.
"""

import time
import serial
import logging

BAUDRATE     = 9600
DEFAULT_PORT = '/dev/ttyS0'
MODULE_IDENTIFIER = "[SerialPort] "
REALLY_SEND_STIM = False

class BiphasicPort(serial.Serial):
    """
    Serial port set up for biphasic pulse communication
    """

    def __init__(self, port=DEFAULT_PORT, baud=BAUDRATE):
        if REALLY_SEND_STIM:
            serial.Serial.__init__(self, port, baud, timeout=0, stopbits=serial.STOPBITS_ONE, \
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE)
            logging.info(MODULE_IDENTIFIER + "Serial port initialized.")

    def sendBiphasicPulse(self):
        if REALLY_SEND_STIM:
            self.setDTR(True)
            time.sleep(0.0002)
            self.setDTR(False)
            time.sleep(0.0001)
            self.setRTS(True)
            time.sleep(0.0002)
            self.setRTS(False)
            time.sleep(0.001)
            logging.info(MODULE_IDENTIFIER + "Biphasic pulse delivered.")
        return
