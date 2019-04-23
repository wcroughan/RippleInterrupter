import logging
import serial
import time

BAUDRATE     = 9600
DEFAULT_PORT = '/dev/ttyS0'

REALLY_STIM = False

class SerialPort(object):

    """
    Serial port connector for sending biphasic pulses using DTR and RTS ports.
    """

    def sendBiphasicPulse(self):
        if not REALLY_STIM:
            return

        self._serial_port.setDTR(True)
        time.sleep(0.0001)
        self._serial_port.setDTR(False)
        time.sleep(0.0001)
        self._serial_port.setRTS(True)
        time.sleep(0.0001)
        self._serial_port.setRTS(False)
        time.sleep(0.001)
        return

    def __init__(self, port_id=DEFAULT_PORT, baudrate=BAUDRATE):
        if not REALLY_STIM:
            return

        # TODO: Put this in a try/catch block
        self._serial_port = serial.Serial(port_id, baudrate, timeout=0, \
                stopbits=serial.STOPBITS_ONE, \
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE)
        return
