"""
Module for maintaining an Adjusting log across recording sessions.
"""

# System Imports
import json
import logging

# Local Imports
import BrainAtlas
import QtHelperUtils

MODULE_IDENTIFIER = "[AdjustingLogger] "

# Adjusting data input is expected to be in #Turns. Converting that into
# distance metric involves a scaling factor (Screw pitch in our case usually.)
TURN_MULTIPLICATION_FACTOR = 0.25

class TetrodeLog(object):

    """
    Class implementing a JSON based data entry system to track tetrode depths.
    This class contains a dictionary in which, each tetrode is mapped to a set of 3 coordinate values
        - [ML] Medial-Lateral
        - [AP] Anterior-Posterior
        - [DV] Dorsal-Ventral
    """

    def __init__(self, tetrode_list, data_file=None):
        self._tetrode_list = tetrode_list
        self._current_placement = dict()

        data_loaded = self.loadDataFile(data_file)
        if not data_loaded:
            print(MODULE_IDENTIFIER + "Starting entries at default value.")
            for t_num in self._tetrode_list:
                self._current_placement[t_num] = [BrainAtlas.DEFAULT_ML_COORDINATE, \
                        BrainAtlas.DEFAULT_AP_COORDINATE, BrainAtlas.DEFAULT_DV_COORDINATE]

    def getCoordinates(self, tetrode):
        """
        Get the current coordinates for a tetrode.
        """
        if tetrode not in self._current_placement:
            logging.warning(MODULE_IDENTIFIER + "Tetrode not found in current placement entry.")
            print(MODULE_IDENTIFIER + "Couldn't find tetrode %d in database"%tetrode)
            # TODO: Maybe add a new entry for this in the future
            return

        return self._current_placement[tetrode]

    def updateDepth(self, tetrode, adjustment):
        """
        Update the depth of a given tetrode by the adjustment amount.
        """
        if tetrode not in self._current_placement:
            logging.warning(MODULE_IDENTIFIER + "Tetrode not found in current placement entry.")
            print(MODULE_IDENTIFIER + "Couldn't find tetrode %d in database"%tetrode)
            # TODO: Maybe add a new entry for this in the future
            return
        initial_tetrode_depth = self._current_placement[tetrode][2]
        self._current_placement[tetrode][2] += adjustment * TURN_MULTIPLICATION_FACTOR
        logging.info(MODULE_IDENTIFIER + "T%d %.2f -> %.2f"%(tetrode, initial_tetrode_depth, \
                self._current_placement[tetrode][2]))
        print(MODULE_IDENTIFIER + "T%d %.2f -> %.2f"%(tetrode, initial_tetrode_depth, \
                self._current_placement[tetrode][2]))

    def writeDataFile(self, output_filename=None):
        """
        Save the current adjustment coordinates to file.
        """
        if output_filename is None:
            output_filename = QtHelperUtils.get_save_file_name(file_format='Adjusting DB (*.json)', \
                    message='Choose Adjusting Database [Save]')

            if not output_filename:
                return

        try:
            with open(output_filename, 'w') as output_file:
                json.dump(self._current_placement, output_file, indent=4, \
                        sort_keys=True, separators=(',', ': '))
                output_file.close()
            logging.info(MODULE_IDENTIFIER + "Log written to %s"output_filename)
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to write data to file.")
            print(err)

    def loadDataFile(self, data_filename):
        data_loaded = False
        if data_filename is None:
            # Ask the user to select a file from which data should be read.
            # Otherwise create a new file
            data_filename = QtHelperUtils.get_open_file_name(file_format='Adjusting DB (*.json)', \
                    message='Choose Adjusting Database [Load]')

            # User can choose not to select a file here.
            if not data_filename:
                return data_loaded

        try:
            with open(data_filename, 'r') as data_file:
                self._current_placement = json.loads(data_file)
            # TODO: Check that all the tetrodes in the current list are there in this database
            data_loaded = True
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to load adjusting data.")
            print(err)

        return data_loaded
