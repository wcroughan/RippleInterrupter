"""
Connection interface to Trodes
"""
from spikegadgets import trodesnetwork as tn

# Constant declaration
LFP_SUBSCRIPTION_ATTRIBUTE = 1024
SPIKE_SUBSCRIPTION_ATTRIBUTE = 1024

class SGClient(tn.AbstractModuleClient):
    """
    Extension of SpikeGadgets client for communicating with Trodes which is
    either recording data in real time or playing back a pre-recorded session.

    William Croughan
    2019/02/11
    """

    timestamps = []
    recvquit = False

    def __init__(self, name, connection="tcp://127.0.0.1", port=49152):
        """
        Pass on some descriptors to the parent class
        """

        # Call the parent class constructor
        tn.AbstractModuleClient.__init__(self, name, connection, port)
        if (self.initialize() != 0):
            raise Exception("Could not connect to Trodes. Aborting!")
        print("Initialized connection to Trodes.")

    def recv_quit(self):
        self.recvquit = True

    def recv_event(self, origin, event, msg):
        if origin == "CameraModule.2" and event == "2Dpos":
            self.recvquit = True

        print(origin)
        print(event)
        print(msg)
