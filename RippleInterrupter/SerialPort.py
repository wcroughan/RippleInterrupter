"""
Module for communicating with serial port for communication.
"""

import time
import serial
import logging

BAUDRATE     = 9600
DEFAULT_PORT = '/dev/ttyS0'
MODULE_IDENTIFIER = "[SerialPort] "

class BiphasicPort(serial.Serial):
    """
    Serial port set up for biphasic pulse communication
    """

    def __init__(self, port=DEFAULT_PORT, baud=BAUDRATE):
        self._is_enabled = FAlse;
        serial.Serial.__init__(self, port, baud, timeout=0, stopbits=serial.STOPBITS_ONE, \
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE)
        logging.info(MODULE_IDENTIFIER + "Serial port initialized.")

    def sendBiphasicPulse(self):
        if self._is_enabled:
            self.setDTR(True)
            time.sleep(0.0002)
            self.setDTR(False)
            time.sleep(0.0001)
            self.setRTS(True)
            time.sleep(0.0002)
            self.setRTS(False)
            time.sleep(0.001)
            logging.info(MODULE_IDENTIFIER + "Biphasic pulse delivered.")
        else:
            logging.info(MODULE_IDENTIFIER + "WARNING! Attempted Biphasic pulse without enabling device! Ignoring!")

    def getStatus(self):
        return self._is_enabled

    def enable(self):
        """
        Enable the serial port (remove pin values from defaults)
        """
        self._is_enabled = True

    def disable(self):
        """
        Disable serial port (allow pin values to be changed by outside input).
        """
        self._is_enabled = False
