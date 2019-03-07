"""
Create configuration structure for different environments with the following fields
    - N_TETRODES: Number of tetrodes which are interesting
    - TETRODE_LIST: List of the tetrode list indices
    - CLUSTER_FILENAME: Name/Location of the cluster file for the data we
      are using. Does not have to be supplied as the program lets you
      choose it using a GUI.

    [OPTIONAL ARGUMENTS]
    TODO: These haven't been added yet.
    - SPEED_THRESHOLD
"""
import os
import logging
import configparser
from tkinter import Tk, filedialog
import xml.etree.ElementTree as ET

DEFAULT_CONFIG_FILE='config/default.ini'
MODULE_IDENTIFIER="[Configuration] "

def readClusterFile(filename=None, tetrodes=None):
    """
    Reads a cluster file and generates a list of tetrodes that have cells and
    all the clusters on that tetrode.

    :filename: XML file containing clustering information.
    :tetrodes: Which tetrodes to look at in the cluster file.
    :returns: A dictionary giving valid cluster indices for each tetrode.
    """
    if filename is None:
        gui_root = Tk()
        gui_root.wm_withdraw()
        filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select Cluster file",filetypes = (("cluster files","*.trodesClusters"),("all files","*.*")))
        gui_root.destroy()

    try:
        cluster_tree = ET.parse(filename)
    except Exception as err:
        print(err)
        return

    # The file is organized as:
    # [ROOT] SpikeSortInfo 
    #       -> PolygonClusters
    #           -> ntrode (nTrodeID)
    #               -> cluster (clusterIndex)

    n_trode_to_cluster_idx_map = {}
    raw_cluster_idx = 0
    # Some unnecessary accesses to get to tetrodes and clusters
    tree_root = cluster_tree.getroot()
    polygon_clusters = tree_root.getchildren()[0]
    ntrode_list = list(polygon_clusters)

    if tetrodes is None:
        tetrodes = range(1,1+len(ntrode_list))

    for ti in tetrodes:
        # Offset by 1 because Trodes tetrodes start with 1!
        ntrode = ntrode_list[ti-1]
        tetrode_idx = ntrode.get('nTrodeIndex')
        if len(list(ntrode)) == 0:
            # Has no clusters on it
            continue

        # TODO: These indices go from 1.. N. Might have to switch to 0.. N if
        # that is what spike data returns.
        cluster_idx_to_id_map = {}
        for cluster in ntrode:
            local_cluster_idx = cluster.get('clusterIndex')
            cluster_idx_to_id_map[int(local_cluster_idx)] = raw_cluster_idx
            raw_cluster_idx += 1
        n_trode_to_cluster_idx_map[ti] = cluster_idx_to_id_map

    # Final value of raw_cluster_idx is a proxy for the total number of units we have
    logging.info(MODULE_IDENTIFIER + "Cluster map...\n%s"% n_trode_to_cluster_idx_map)
    return raw_cluster_idx, n_trode_to_cluster_idx_map

def get_open_field_configuration(filename=None):
    """
    Return configuration for the open field.
    """
    if filename is None:
        gui_root = Tk()
        gui_root.wm_withdraw()
        filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select Cluster file",filetypes = (("cluster files","*.trodesClusters"),("all files","*.*")))

    configuration = configparser.ConfigParser()
    try:
        configuration.read(filename)
    except Exception as err:
        print('Unable to read configuration file %s. Using defaults.'%filename)
        configuration.read(DEFAULT_CONFIG_FILE)
