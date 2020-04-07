from IFU import *
import configparser
import numpy as np
import sys
import os
        
        
def read_ini(section):
    arg = {}
    for key in section:
        value = section[key]
        if '.' in value:
            arg[key] = float(value)
        else:
            arg[key] = int(value)
    return arg


config = configparser.ConfigParser()
config.read('scaled.ini')

arg = {}
arg.update(read_ini(config['Defined']))
arg.update(read_ini(config['Dervied']))
arg.update(read_ini(config['User']))     

def scales(arg, sr, mag):
    
    smtmt = TMTPupil()
    smtmt.load_from('smtmt.fits')
    min_wl = arg['min_wavelength']
    max_wl = arg['max_wavelength']
    n_wl = arg['no_spaxel']
    wavelen = np.linspace(min_wl, max_wl, n_wl) * 1e-6
    dwl = (max_wl - min_wl)/n_wl * 1e3

    #parameters
    #envo
    vapor = 1
    airmass = 1.5
    temp = -15
    RH = 10
    P = 700
    SR = sr
    
    #target
    mag = mag
    
    #IFU mode
    
    
    #calculate
    fresh_template = False

    mirrors = [0.95, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99, 0.95, 0.95]
    IFUmirrors = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99]
    temp1 = -15 + 273.16
    temp2 = 0
    
    
    temps = [temp1 for m in mirrors]
    temps.extend([temp2 for m in IFUmirrors])
    allmirror = mirrors + IFUmirrors

    ps = PhaseScreen(pupil = smtmt, strehl = {'value': SR, 'wl': 3.5e-6})
    ps.type = 'SR%02d'%(sr*100)
    st = SkyTrans(vapor, airmass)
    sbg = SkyBG(vapor, airmass)
    disp = Dispersion(RH, temp, P, airmass)
    ite = InstTransEm(allmirror, temps)
    phase = {'type':ps.type, 'screen':ps.get_res()}

    ccd = NoiseCamera(smtmt)

    fov = 1.2
    ccd.pixsize = 180
    ccd.pixscal = fov/ccd.pixsize
    ccd.factor = 2
    ccd.template = 'temp/SR%02d_temp.fits'%(sr*100)
    ccd.save_size = 360
    
    ccd.save_template = 'Y'
    if os.path.exists(ccd.template):
        print('load')
        ccd.save_template = 'N'  #auto
        ccd.load_template = 'Y'
    
    if fresh_template:
        ccd.save_template = 'Y'
        ccd.load_template = 'N'
        
    print(ccd.save_template)
    
    #target 

    tar1 = Target(pos = [0,0])
    tar1.Temp = 5000
    tar1.set_mag(mag, 1e-6)
    
    # tar2 = Target(pos = [0.1, 0])
    # tar2.Temp = 2000
    # tar2.set_mag(mag + 5, 1e-6)
    
    #observation
    
    exp = 10
    psf,noise = ccd.get_image(wavelen, exp, [tar1], phase, st, sbg, disp, ite, dwl)
    
    name = 'sr%03dm%02d'%(100*sr, mag)
    img = ccd.make_noise(psf + noise)
    
    fits.writeto('res/' + name + '_psf.fits', psf)
    fits.writeto('res/' + name + '_noi.fits', noise)
    fits.writeto('res/' + name + '_img.fits', img)
        
    # ifu = Lenslet(arg)
    # ifusig = ifu.run_fast(wavelen, psf + noise)
    # ifu_f = ccd.make_noise(ifusig)
    #
    # hd = fits.Header()
    # hd['strehl'] = sr
    # hd['vapor'] = vapor
    # hd['airmass'] = airmass
    # hd['temp'] = temp
    # hd['RH'] = RH
    # hd['P'] = P
    # hd['minWave'] = min_wl
    # hd['maxWave'] = max_wl
    # hd['target1'] = mag
    # hd['exp'] = exp
    #
    # #SNR
    #
    # fits.writeto('res/ifu%02d.fits'%(sr*100), ifu_f, hd,  overwrite = True)
    
#     ifunoise = ifu.run(wavelen, noise)
#     fits.writeto('ifunoise.fits',ifunoise,overwrite = True)

scales(arg, float(sys.argv[1]), float(sys.argv[2]))