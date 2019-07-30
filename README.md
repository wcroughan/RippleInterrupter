# OnlineAnalysis
Applications and scripts for interacting with Trodes. This application was developed with the intent of running online ripple interruption. With some use, however, it seemed that the application could also be used as a helper tool for adjusting.

# Requirements
Trodes V2.0 (or higher): This supports the python libraries that can read data from Trodes continuously
python3.7 with the following libraries installed:
 - spikegadgets: Python library that works with Trodes
 - PyQt5
 - numpy
 - scipy
 - matplotlib
 
 # How to run
 Launch the main file (OnlineInterruption.py) from the commandline as
  $ python -O OnlineInterruption.py
The app can be launched with the -O option but that adds significant debugging overheads (and generated a lot of log files). It will be useful if something is not working correctly and you want a more detailed explaination for why things are going wrong.

Feel free to report bugs in the app!
