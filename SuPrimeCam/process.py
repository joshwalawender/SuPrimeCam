#!/usr/env/python

## Import General Tools
from pathlib import Path
import argparse
import logging



##-------------------------------------------------------------------------
## Parse Command Line Arguments
##-------------------------------------------------------------------------
## create a parser object for understanding command-line arguments
p = argparse.ArgumentParser(description='''
''')
## add flags
p.add_argument("-v", "--verbose", dest="verbose",
    default=False, action="store_true",
    help="Be verbose! (default = False)")
## add options
p.add_argument("--input", dest="input", type=str,
    help="The input.")
## add arguments
p.add_argument('argument', type=int,
               help="A single argument")
p.add_argument('allothers', nargs='*',
               help="All other arguments")
args = p.parse_args()


##-------------------------------------------------------------------------
## Create logger object
##-------------------------------------------------------------------------
log = logging.getLogger('MyLogger')
log.setLevel(logging.DEBUG)
## Set up console output
LogConsoleHandler = logging.StreamHandler()
LogConsoleHandler.setLevel(logging.DEBUG)
LogFormat = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
LogConsoleHandler.setFormatter(LogFormat)
log.addHandler(LogConsoleHandler)
## Set up file output
# LogFileName = None
# LogFileHandler = logging.FileHandler(LogFileName)
# LogFileHandler.setLevel(logging.DEBUG)
# LogFileHandler.setFormatter(LogFormat)
# log.addHandler(LogFileHandler)


##-------------------------------------------------------------------------
## Main Program
##-------------------------------------------------------------------------
def process(MEF40path):
    MEF40path = Path(MEF40path).expanduser()
    files = MEF40path.glob('*.fits')

    


if __name__ == '__main__':
    MEF40path = Path('/Volumes/ScienceData/SuPrimeCam/SuPrimeCam_S17A-UH16A/Processed')

    process(MEF40path)
