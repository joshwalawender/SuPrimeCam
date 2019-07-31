#!/usr/env/python

## Import General Tools
from pathlib import Path
import logging

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table

import ccdproc

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
## Exceptions
##-------------------------------------------------------------------------
class MEFDataError(Exception):
    """Base class for exceptions in this module."""
    pass


class IncompatiblePixelData(MEFDataError):
    """Raise when trying to operate on multiple MEFData
    objects which have incompatible pixeldata.
    """
    def __init__(self, message):
        super().__init__(f"MEFData objects have incompatible pixeldata. {message}")


class IncorrectNumberOfExtensions(MEFDataError):
    """Raise when verify method fails for a specific instrument.
    """
    def __init__(self, datatype, expected, kd):
        msg = f"Incorrect number of {datatype} entries.  Expected {expected} for {type(kd)}"
        print(msg)
        super().__init__(msg)


##-------------------------------------------------------------------------
## MEFData Classes
##-------------------------------------------------------------------------
class MEFData(object):
    """Our data model.
    
    Attributes:
    pixeldata -- a list of CCDData objects containing pixel values.
    tabledata -- a list of astropy.table.Table objects
    headers -- a list of astropy.io.fits.Header objects
    """
    def __init__(self, *args, **kwargs):
        self.pixeldata = []
        self.tabledata = []
        self.headers = []

    def verify(self):
        """Method to check the data against expectations. For the 
        MEFData class this simply passes and does nothing, but
        subclasses for specific instruments can populate this
        with appropriate tests.
        """
        pass

    def add(self, kd2):
        """Method to add another MEFData object to this one and return
        the result.  This uses the CCDData object's add method and
        simply loops over all elements of the pixeldata list.
        """
        if len(self.pixeldata) != len(kd2.pixeldata):
            raise IncompatiblePixelData
        for i,pd in enumerate(self.pixeldata):
            self.pixeldata[i] = pd.add(kd2.pixeldata[i])

    def subtract(self, kd2):
        """Method to subtract another MEFData object to this one
        and return the result.  This uses the CCDData object's
        subtract method and simply loops over all elements of
        the pixeldata list.
        """
        if len(self.pixeldata) != len(kd2.pixeldata):
            raise IncompatiblePixelData
        for i,pd in enumerate(self.pixeldata):
            self.pixeldata[i] = pd.subtract(kd2.pixeldata[i])

    def multiply(self, kd2):
        """Method to multiply another MEFData object by this one
        and return the result.  This uses the CCDData object's
        multiply method and simply loops over all elements of
        the pixeldata list.
        """
        if len(self.pixeldata) != len(kd2.pixeldata):
            raise IncompatiblePixelData
        for i,pd in enumerate(self.pixeldata):
            self.pixeldata[i] = pd.multiply(kd2.pixeldata[i])

    def get(self, kw):
        """Method to loop over all headers and get the specified keyword value.
        Returns the first result it finds and doe not check for duplicate
        instances of the keyword in subsequent headers.
        """
        for hdr in self.headers:
            val = hdr.get(kw, None)
            if val is not None:
                return val

    def create_deviation(self, readnoise=10):
        for i,pd in enumerate(self.pixeldata):
            gain = pd.header.get('GAIN')
            self.pixeldata[i] = ccdproc.create_deviation(
                pd, gain=gain * u.electron/u.adu,
                readnoise=readnoise * u.electron)

    def gain_correct(self):
        for i,pd in enumerate(self.pixeldata):
            gain = pd.header.get('GAIN')
            self.pixeldata[i] = ccdproc.gain_correct(pd, gain*u.electron/u.adu)

    def bias_subtract(self, master_bias):
        for i,pd in enumerate(self.pixeldata):
            self.pixeldata[i] = ccdproc.subtract_bias(pd, master_bias)

    def write(self):
        '''Assemble in to chips and write as 10 extension MEF.
        '''
        for chip in range(0,10):
            ext0 = chip*4
            x0s = []
            x1s = []
            y0s = []
            y1s = []
            for ext in range(chip*4, (chip+1)*4):
                hdr = self.headers[ext+1]
                assert hdr.get('DET-ID') == chip
                detsec = hdr.get('DETSEC')[1:-1]
                x0, x1 = detsec.split(',')[0].split(':')
                y0, y1 = detsec.split(',')[1].split(':')
                x0s.append(int(x0))
                x1s.append(int(x1))
                y0s.append(int(y0))
                y1s.append(int(y1))
            chip_xrange = [min(x0s), max(x1s)]
            chip_yrange = [min(y0s), max(y1s)]
            chip_size = [max(x1s)-min(x0s), max(y1s)-min(y0s)]
            chip_x0s = [x-min(x0s) for x in x0s]
            chip_x1s = [x-min(x0s) for x in x1s]
#             print(chip, chip_x0s)
#             print(chip, chip_x1s)
#             print(chip, y0s)
#             print(chip, y1s)
#             print(chip_size)

            chip_data = np.zeros((chip_size[1]+1, chip_size[0]+1))
            for i,ext in enumerate(range(chip*4, (chip+1)*4)):
                print(i, ext)
                chip_data[:,chip_x0s[i]:chip_x1s[i]+1] = self.pixeldata[ext].data

            chip_hdu = fits.ImageHDU(chip_data, self.headers[ext0+1])
#             print(chip_hdu.data.shape)
#             print(chip_hdu)

            from matplotlib import pyplot as plt
            from astropy.visualization import PercentileInterval, ImageNormalize
            norm = ImageNormalize(chip_hdu.data, interval=PercentileInterval(98))
            plt.figure(figsize=(5,15))
            plt.imshow(chip_hdu.data, origin='lower', norm=norm)
            plt.show()



##-------------------------------------------------------------------------
## Get HDU Type
##-------------------------------------------------------------------------
def get_hdu_type(hdu):
    """Function to examine a FITS HDU object and determine its type.  Returns
    one of the following strings:
    
    'header' -- This is a PrimaryHDU or ImageHDU with no pixel data.
    'pixeldata' -- This is a PrimaryHDU or ImageHDU containing pixel data.
    'uncertainty' -- This is a pixeldata HDU which is associated with the
                     uncertainty data written by either CCDData or MEFData.
    'mask' -- This is a pixeldata HDU which is associated with the mask
              data written by either CCDData or MEFData.
    'tabledata' -- This is a TableHDU type HDU.
    """
    if type(hdu) in [fits.PrimaryHDU, fits.ImageHDU] and hdu.data is None:
        # This is a header only HDU
        return 'header'
    elif type(hdu) in [fits.PrimaryHDU, fits.ImageHDU] and hdu.data is not None:
        # This is a pixel data HDU
        extname = hdu.header.get('EXTNAME', '').strip()
        if extname == 'MASK':
            # This is a mask HDU
            return 'mask'
        elif extname == 'UNCERT':
            # This is an uncertainty HDU
            return 'uncertainty'
        else:
            # This must be pixel data
            return 'pixeldata'
    elif type(hdu) == fits.TableHDU:
            # This is table data
            return 'tabledata'


##-------------------------------------------------------------------------
## MEFData Reader
##-------------------------------------------------------------------------
def fits_MEFdata_reader(file, defaultunit='adu', datatype=MEFData):
    """A reader for MEFData objects.
    
    Currently this is a separate function, but should probably be
    registered as a reader similar to fits_ccddata_reader.
    
    Arguments:
    file -- The filename (or pathlib.Path) of the FITS file to open.
    Keyword arguments:
    defaultunit -- If the BUNIT keyword is unable to be located or
                   parsed, the reader will assume this unit.  Defaults
                   to "adu".
    datatype -- The output datatype.  Defaults to MEFData, but could
                be a subclass such as MOSFIREData.  The main effect of
                this is that it runs the appropriate verify method on
                the data.
    """
    try:
        hdul = fits.open(file, 'readonly')
    except FileNotFoundError as e:
        print(e.msg)
        raise e
    except OSError as e:
        print(e.msg)
        raise e
    # Loop though HDUs and read them in as pixel data or table data
    md = datatype()
    while len(hdul) > 0:
#         print('Extracting HDU')
        hdu = hdul.pop(0)
        md.headers.append(hdu.header)
        hdu_type = get_hdu_type(hdu)
#         print(f'  Got HDU type = {hdu_type}')
        if hdu_type == 'header':
            pass
        elif hdu_type == 'tabledata':
            md.tabledata.append(Table(hdu.data))
        elif hdu_type == 'pixeldata':
            # Check the next HDU
            mask = None
            uncertainty = None
            if len(hdul) > 0:
                next_type = get_hdu_type(hdul[0])
                if next_type == 'mask':
                    mask = hdul[0].data
                elif next_type == 'uncertainty':
                    uncertainty = hdul[0].data
            if len(hdul) > 1:
                next_type2 = get_hdu_type(hdul[1])
                if next_type2 == 'mask':
                    mask = hdul[1].data
                elif next_type2 == 'uncertainty':
                    uncertainty = hdul[1].data               
            # Sanitize "ADU per coadd" BUNIT value
            if hdu.header.get('BUNIT') == "ADU per coadd":
                hdu.header.set('BUNIT', 'adu')
            # Populate the CCDData object
            c = CCDData(hdu.data, mask=mask, uncertainty=uncertainty,
                        meta=hdu.header,
                        unit=(hdu.header.get('BUNIT', defaultunit)).lower(),
                       )
            md.pixeldata.append(c)
#     print(f'Read in {len(md.headers)} headers, '
#           f'{len(md.pixeldata)} sets of pixel data, '
#           f'and {len(md.tabledata)} tables')
    md.verify()
    return md


##-------------------------------------------------------------------------
## Main Program
##-------------------------------------------------------------------------
def process(MEF40path):
    MEF40path = Path(MEF40path).expanduser()

    # Create table of files
    tablefile = MEF40path.parent.joinpath('files.txt')
    if tablefile.exists() is True:
        print(f"Reading {tablefile}")
        t = Table.read(tablefile, format='ascii.csv')
    else:
        t = Table(names=('file', 'imtype', 'filter', 'object'),
                  dtype=('a200',  'a12', 'a12', 'a50') )
        for file in MEF40path.glob('MEF*.fits'):
            MEF = fits_MEFdata_reader(file)
            t.add_row((file, MEF.get('DATA-TYP'), MEF.get('FILTER01'),
                       MEF.get('OBJECT')))
        t.write(tablefile, format='ascii.csv')

#     print(t)

    print('Testing write functionality')
    images_i = t[(t['imtype'] == 'OBJECT')
               & (t['filter'] == 'W-S-I+')
               & (t['object'] == 'RhoOph1')]
    file = images_i[0]['file']
    print(f"  Using: {file}")
    MEF = fits_MEFdata_reader(file)
    MEF.write()
    import sys ; sys.exit(0)

    ##-------------------------------------------------------------------------
    ## Build Master Bias
    ##-------------------------------------------------------------------------
    biases = t[t['imtype'] == 'BIAS']
    print(f'Processing {len(biases)} BIAS files')
    bias_MEFs = []
    for file in biases['file']:
        print(f'  Reading {file}')
        MEF = fits_MEFdata_reader(file)
        print(f'  Creating deviation')
        MEF.create_deviation()
        print(f'  Gain correcting')
        MEF.gain_correct()
        bias_MEFs.append(MEF)

    master_bias = bias_MEFs[0]
    for i,pd in enumerate(master_bias.pixeldata):
        pds = [bias.pixeldata[i] for bias in bias_MEFs]
        master_bias.pixeldata[i] = ccdproc.combine(pds, method='average',
            clip_extrema=True, nlow=1, nhigh=1)

    print(master_bias)

    ##-------------------------------------------------------------------------
    ## Build Master DOMEFLAT (i filter)
    ##-------------------------------------------------------------------------
    domeflats_i = t[(t['imtype'] == 'DOMEFLAT') & (t['filter'] == 'W-S-I+')]
    print(f'Processing {len(domeflats)} DOMEFLAT files in the i filter')
    domeflat_i_MEFs = []
    for file in domeflats_i['file']:
        print(f'  Reading {file}')
        MEF = fits_MEFdata_reader(file)
        print(f'  Creating deviation')
        MEF.create_deviation()
        print(f'  Gain correcting')
        MEF.gain_correct()
        print(f'  Bias subtracting')
        MEF.bias_subtract(master_bias)
        domeflat_i_MEFs.append(MEF)

    master_flat_i = domeflat_i_MEFs[0]
    for i,pd in enumerate(master_flat_i.pixeldata):
        pds = [im.pixeldata[i] for im in domeflat_i_MEFs]
        
        master_flat_i.pixeldata[i] = ccdproc.combine(pds, method='average',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            scale=np.median)

    print(master_flat_i)

    ##-------------------------------------------------------------------------
    ## Process Science Frames (i filter)
    ##-------------------------------------------------------------------------
    images_i = t[(t['imtype'] == 'OBJECT') & (t['filter'] == 'W-S-I+')]
    print(f'Processing {len(images_i)} OBJECT files in the i filter')
    images_i_MEFs = []
    for file in images_i['file']:
        print(f'  Reading {file}')
        MEF = fits_MEFdata_reader(file)
        print(f'  Creating deviation')
        MEF.create_deviation()
        print(f'  Gain correcting')
        MEF.gain_correct()
        print(f'  Bias subtracting')
        MEF.bias_subtract(master_bias)
        images_i_MEFs.append(MEF)



if __name__ == '__main__':
    MEF40path = Path('/Volumes/ScienceData/SuPrimeCam/SuPrimeCam_S17A-UH16A/Processed/MEF')

    process(MEF40path)
