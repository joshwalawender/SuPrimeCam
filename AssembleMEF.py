import os
import sys
from pathlib import Path

from glob import glob
from astropy import units as u
from astropy.io import fits
import numpy as np

def assemble_MEF(file0, outdir=None):
    if outdir is not None:
        outdir = os.path.expanduser(outdir)
        assert os.path.exists(outdir)

    rawdir, filename0 = os.path.split(file0)
    files = ['{}{:d}.fits'.format(filename0[0:11], n) for n in range(0,10)]
    rawhduls = [fits.open(os.path.join(rawdir, file), 'readonly') for file in files]
    expid0 = rawhduls[0][0].header.get('EXP-ID')

    # Confirm that EXP-IDs match indicating that these files are from the same exposure
    for hdul in rawhduls:
        expid = hdul[0].header.get('EXP-ID')
        if not expid == expid0:
            print(f'WARNING: EXP-ID mismatch!! {expid0}, {expid}')
    
    # Split each file out in to channels and assemble a multi extension fits (MEF)
    # file from the individual amplifiers (4 per chip)
    MEF = fits.HDUList([])
    extver = 0
    for i,hdul in enumerate(rawhduls):
        detid = hdul[0].header.get('DET-ID')
        detx = hdul[0].header.get('DET-P101')
        dety = hdul[0].header.get('DET-P201')
        chxpos = []
        chypos = []
        minyos = {}
        minyef = {}
        minxos = {}
        minxef = {}
        maxyos = {}
        maxyef = {}
        maxxos = {}
        maxxef = {}

        for ch in [1,2,3,4]:
            minxos[ch] = hdul[0].header.get(f'S_OSMN{ch}1')
            maxxos[ch] = hdul[0].header.get(f'S_OSMX{ch}1')
            minyos[ch] = hdul[0].header.get(f'S_OSMN{ch}2')
            maxyos[ch] = hdul[0].header.get(f'S_OSMX{ch}2')

            minxef[ch] = hdul[0].header.get(f'S_EFMN{ch}1')
            maxxef[ch] = hdul[0].header.get(f'S_EFMX{ch}1')
            minyef[ch] = hdul[0].header.get(f'S_EFMN{ch}2')
            maxyef[ch] = hdul[0].header.get(f'S_EFMX{ch}2')
            chxpos.append(minxef[ch])
            chypos.append(minyef[ch])

        minxpix = min(chxpos)
        minypix = min(chypos)
        tb = {True: 2, False: 1}[dety > -30]
        DETSECy2 = tb * (maxyef[1] - minypix)
        DETSECy1 = DETSECy2 - (maxyef[1] - minyef[1]) + 1

        for ch in [1,2,3,4]:
            extver += 1
#             miny = min([minyos[ch], minyef[ch]])
#             minx = min([minxos[ch], minxef[ch]])
#             maxy = max([maxyos[ch], maxyef[ch]])
#             maxx = max([maxxos[ch], maxxef[ch]])
#             chdata = hdul[0].data[miny:maxy,minx:maxx]
            chdata = hdul[0].data[minyef[ch]-1:maxyef[ch],minxef[ch]-1:maxxef[ch]]
            if extver == 1:
                phdu = fits.PrimaryHDU(None, hdul[0].header)
                MEF.append(phdu)
            chhdu = fits.ImageHDU(chdata, hdul[0].header, name=f'd{detid}c{ch}')
            chhdu.header.set('GAIN', hdul[0].header.get(f'S_GAIN{ch}'),
                             f"AD conversion factor (electron/ADU) for ch{ch}")
            chhdu.header.set('EXTNAME', f'im{extver}')
            chhdu.header.set('EXTVER', extver)
            chhdu.header.set('IMAGEID', f'd{detid}c{ch}')
            chhdu.header.set('CCDNAME', hdul[0].header.get('DETECTOR'))
            binx = int(hdul[0].header.get('BIN-FCT1'))
            biny = int(hdul[0].header.get('BIN-FCT2'))
            chhdu.header.set('CCDSUM', f'{binx:d} {biny:d}')
            chhdu.header.set('FILTER', hdul[0].header.get('FILTER01'))
            # OBSTYPE: "zero", "dark", flat", and "object"
            obstype_trans = {'DOMEFLAT': 'flat', 'SKYFLAT': 'flat',
                             'BIAS': 'zero', 'OBJECT': 'object', 'DARK': 'dark'}
            hdrtype = hdul[0].header.get('DATA-TYP').strip()
            if hdrtype not in obstype_trans.keys():
                print(f'WARNING: Could not translate data type for {hdrtype}.')
                print(f'         Skipping file {filename0}')
                return None
            obstype = obstype_trans[hdrtype]
            chhdu.header.set('OBSTYPE', obstype)

            DATASECx1 = 1
            DATASECx2 = maxxef[ch] - minxef[ch]
            DATASECy1 = 1
            DATASECy2 = maxyef[ch] - minyef[ch]
            datasec = f'[{DATASECx1:d}:{DATASECx2:d},{DATASECy1:d}:{DATASECy2:d}]'
            chhdu.header.set('DATASEC', datasec)

            detsecy = {True: 2, False: 1}[dety > 0]
            ampwidth = maxxef[ch] - minxef[ch]
            chipwidth = 4 * ampwidth
            if detx > 30:
                DETSECx1 = 4 * chipwidth + (maxxef[ch] - ampwidth - minxef[ch] % ampwidth) + 1
            elif detx > 0:
                DETSECx1 = 3 * chipwidth + (maxxef[ch] - ampwidth - minxef[ch] % ampwidth) + 1
            elif detx > -30:
                DETSECx1 = 2 * chipwidth + (maxxef[ch] - ampwidth - minxef[ch] % ampwidth) + 1
            elif detx > -60:
                DETSECx1 = 1 * chipwidth + (maxxef[ch] - ampwidth - minxef[ch] % ampwidth) + 1
            else:
                DETSECx1 = 0 * chipwidth + (maxxef[ch] - ampwidth - minxef[ch] % ampwidth) + 1
            DETSECx2 = DETSECx1 + ampwidth -1
            chhdu.header.set('DETSEC', f'[{DETSECx1:d}:{DETSECx2:d},{DETSECy1:d}:{DETSECy2:d}]')
            MEF.append(chhdu)

            CCDSECx1 = 1
            CCDSECx2 = DETSECx2 - DETSECx1 + 1
            CCDSECy1 = 1
            CCDSECy2 = DETSECy2 - DETSECy1 + 1
            ccdsec = f'[{CCDSECx1:d}:{CCDSECx2:d},{CCDSECy1:d}:{CCDSECy2:d}]'
            chhdu.header.set('CCDSEC', ccdsec)



    if outdir is not None:
        outfile = os.path.join(outdir, filename0.replace('SUPA', 'MEF_'))
        MEF.writeto(outfile, overwrite=True)
    return MEF
    
if __name__ == '__main__':
    raw0_path = Path('/Volumes/ScienceData/SuPrimeCam/SuPrimeCam_S17A-UH16A/o16308/')
    raw0_files = [f for f in raw0_path.glob('SUPA*0.fits')]

    print(f"Found {len(raw0_files)} SuPrimeCam images.")

    for file in raw0_files:
        print(f'Assembling MEF for {file.name}')
        assemble_MEF(file, outdir='/Volumes/ScienceData/SuPrimeCam/SuPrimeCam_S17A-UH16A/Processed/MEF')

