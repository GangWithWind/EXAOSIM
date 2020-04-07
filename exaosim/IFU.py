from astropy.io import fits
import scipy.ndimage as nd
import numpy as np
from math import *
import matplotlib.pyplot as plt
import numpy.random as nprand
import ifu_patch

print('hi')

#===target====
class Target(object):
    '''Spectrum of target.
    
    attributes:
        None
    methods:
        set_mag
        get_res

    '''
    def __init__(self, pos = [0,0], temp = 5700):
        '''
        input:
            None
        keyword:
            (float, float) pos = [0, 0]: position of the target in the Fov, 
            unit in arcsec. [0, 0] means the center of FOV
        
            (float) temp = 5700: Temperature of the object
        
        '''
        self.Temp = temp
        self.type = 'BB'
        self.pos = np.array(pos)
        self.vega = np.genfromtxt('./data/vega.dat')
        self.wl_ref = 2e-6
        
    def vega_ref(self, wavelen):
        w = np.interp(wavelen*1e6, self.vega[:,0], self.vega[:,2])
        h = 6.62607004e-34
        w_photon = h*3e8/wavelen
        return w / w_photon
    
    def set_mag(self, mag, wavelen, ttype = 'BB'):
        '''Set the magnitude of the object.
        input: 
            (float) mag: magnitude
            (float) wavelen: wavelength of the magnitude
        
        keyword:
            (float) ttype = 'BB': target type, 'BB' for black body, 'RJ' for Rayleigh–Jeans
        
        return:
            None
        
        '''
        f_vega = self.vega_ref(wavelen)      
        f_star = 10**(-mag/2.5) * f_vega
        
        if ttype == 'BB':
            f_black = self.blackbody(wavelen)
            self.type = ttype
        elif ttype == 'RJ':
            f_black = self.RJ(wavelen)
            self.type = 'RJ'
            
        self.mag = mag
        self.factor = f_star/f_black
        self.wl_ref = wavelen
        
        
    def set_Jy(self, Jy, ttype = 'BB'):
        self.Jy = Jy
        
        
    
    def RJ_old(self, wl):
        hconst = 6.62607004e-34 #J/s
        kconst = 1.3806488e-23 #J/K
        splight = 3e8
        SBconst = 5.670373e-8
        totalL = SBconst * self.Temp ** 4
        core = hconst * splight / wl / kconst / self.Temp
        res = 2 / totalL * splight * np.pi / wl ** 4 / core
        return res
        
    def RJ(self, wl):
        # hconst = 6.62607004e-34 #J/s
        # kconst = 1.3806488e-23 #J/K
        # splight = 3e8
        # SBconst = 5.670373e-8
        # totalL = SBconst * self.Temp ** 4
        # core = hconst * splight / wl / kconst / self.Temp
        # res = 2 / totalL * splight * np.pi / wl ** 4 / core
        
        cwl = wl[len(wl) // 2] 
        flux = self.Jy * 1.51e7 / (1/cwl)
        core = (wl/cwl)**-3 * flux
        return res
    
    def blackbody(self, wl):
        hconst = 6.62607004e-34 #J/s
        kconst = 1.3806488e-23 #J/K
        splight = 3e8
        SBconst = 5.670373e-8
        totalL = SBconst * self.Temp ** 4
        core = np.exp(hconst * splight / wl / kconst / self.Temp) - 1
        res = 2 * splight * np.pi / wl ** 4 / core
        return res
    
    def power_law(self, wl, n):
        hconst = 6.62607004e-34 #J/s
        kconst = 1.3806488e-23 #J/K
        L0 = 2000e-9
        splight = 3e8
        SBconst = 5.670373e-8
        totalL = SBconst * self.Temp ** 4
        core = hconst * splight / kconst / self.Temp
        res = (n - 1) * self.flux / totalL * splight * np.pi / wl ** n / core / L0 ** (3-n)
        return res
    
    def get_res(self, wl):
        '''return photon flux at the give wavelength.
        input:
            (np.array) wl: input wavelength
        
        return: 
            (np.array): photon flux at the give wavelength.
        '''
        if self.type == 'BB':
            return self.blackbody(wl) * self.factor
        elif self.type == 'RJ':
            return self.RJ(wl)

#==================basic functions and classes========================
def primefactors(n):
    '''Generate all prime factors of n. used in the camera modula to find the fastest fft size.
    input: (int) n
    output: (int) number of prime factor of n
    '''
    f = 2
    nres = 0
    while f * f <= n:
        while not n % f:
            nres += 1
            n //= f
        f += 1
    if n > 1:
        nres += 1
    return nres

def rebin(a, factor):
    '''binning of an image, can only do n*n binning
    input: 
        (np.array) a : image
        (int) factor : factor of binning
    output:
        (np.array) : image after binning
    '''
    shape = a.shape
    binshape = (np.asarray(shape)/factor).astype(int)
    subshape = binshape * factor
    sub = a[0:subshape[0],0:subshape[1]]
    return sub.reshape(binshape[0],factor,binshape[0],factor).mean(1).mean(2)

def reshape(psf,rate,anti_aliasing = 0):
    '''
    zoom in and zoom out base on the center of the image. using order 3 spline interpolation.
    this function is used in the Camera Object, to adjust the pixel rate. 
    the size of image will not change.
    
    input:  
        (np.array) psf: input image
        (float) rate: zoom rate, >1, zoom out, <1, zoom in.
    keywords:
        (int) anti_aliasing: method for anti_aliasing (not finished)
    output:  
        (np.array): zoomed image. 
    '''
    if anti_aliasing == 1:
        fre = np.fft.fftshift(np.fft.fft2(psf))
        dx = int(sz[0] * (1 - rate) /2)
        fre[:dx,0] = 0
        fre[-dx:,0] = 0
        fre[:0,dx] = 0
        fre[0:-dx:] = 0
        psf = abs(np.fft.ifft2(fre))
     
    if anti_aliasing == 2:
        xbin = 1
        if rate >= 0.5:
            xbin = 2
        psf = rebin(psf,xbin)
        rate /= xbin

    sz = psf.shape
        
    def coor_change(coor_in):
        out0 = coor_in[0] * rate + ((sz[0] - 1) / 2) * (1 - rate)
        out1 = coor_in[1] * rate + ((sz[1] - 1) / 2) * (1 - rate)
        return (out0,out1)
    res=nd.interpolation.geometric_transform(psf,coor_change)
    return res
    
class DataFile(object):
    '''
    Basic object for the data which need to be loaded from data file. 
    The data file should have Independent variable in the first column
    and depnedent variables in other columns.
    
    input: 
        None
    attributes:
        res_factor: result = result * res_factor, used to change unit of output
        wl_factor: used to change unit of input wavelength, need to be modified by sub-object
    method:
        get_res
    '''
    
    def __init__(self):
        self.filename = ''
        self.default_name = ''
        self.wl_factor = 1e9
        self.out = None
        self.res_factor = 1
        
    def get_data(self):
        try:
            self.data = np.genfromtxt('./data/%s'%self.filename)
        except:
            print('can not load file: {}'.format(self.filename))
            print('using default file: {}'.format(self.default_name))
            self.filename = self.default_name
            self.data = np.genfromtxt('./data/%s'%self.default_name)
        self.data[:, 0] /= self.wl_factor
        
    def average_interval(self, wavelength, datax, datay, interp = False, dout = None):
    
        if interp:
            return np.interp(wavelength, datax, datay)

        wavelength = np.sort(wavelength)
        wmin = wavelength[0]
        wmax = wavelength[-1]
        if not dout:
            if wmax == wmin:
                raise IndexError('wavelength must have more than 2 different elements if dout is not set')
            dout = (wmax - wmin) / (len(wavelength) - 1)
            
        ind = 0
        nx = len(datax)
        
        xres = np.zeros(len(wavelength))
        for iwl, wl in enumerate(wavelength):
            wlnow = []
            for i in range(ind, nx):
                if datax[i] > wl + dout / 2:
                    ind = i
                    if wlnow:
                        xres[iwl] = np.mean(wlnow)
                    else:
                        xres[iwl] = 0
                    break
                if i == nx - 1:
                    ind = nx
                    if wlnow:
                        xres[iwl] = np.mean(wlnow)
                    else:
                        xres[iwl] = 0
                    break
                if datax[i] > wl - dout / 2:
                    wlnow.append(datay[i])

        return xres
        
    def get_res(self, wavelength, out_col = None, dout = None, interp = False):
        '''find value at given wavelength
        
        input:
            (np.array): wavelength
        keywords:
            (list) out_col = []: index of output columns
            (float) dout = None: bandwidth, None menas bandwidth is distance between any two input wavelength
        output:
             (np.array): output by integrate on the bandwidth
        
        '''
        if not out_col:
            out_col = list(range(1, self.data.shape[1]))

        if len(out_col) == 1:
            res = self.average_interval(wavelength, self.data[:, 0], self.data[:, out_col[-1]], dout = dout)
            return res * self.res_factor
            
        res = np.zeros((len(wavelength), len(out_col)), dtype = float)
        for i,icol in enumerate(out_col):
            res[:, i] = self.average_interval(wavelength, self.data[:, 0], self.data[:, icol], dout = dout) 
        return res * self.res_factor

#=================sky model=====================
class PhaseScreen(object):
    '''generate phase screen 
    method: 
        get_res
    '''
    def __init__(self, r0=0.1, L0=None, pixsize=None, 
        sizem=None, pupil = None, strehl = None): 
        '''
        input:
            None
        keyword:
            (float) r0 = 0.1: fred paramener in meter
            (float) L0 = None: outer scale in meter, None for infinity
            (float) pixsize = None: output pixel size of the phase screen
            (float) sizem = None: phyical size of the phase screen
            (bool) pupil = None: pupil object, pixsize and sizem will copy from pupil if it is set. 
            (float) strehl = None: re-scale RMS of the phase due to the stehl ratio of the image.
        '''
        self.pixsize = pixsize
        self.sizem = sizem
        
        self.r0 = r0
        self.uncorelated = True
        self.L0 = L0
        self.ref = 500e-9
        self.sr_factor = np.nan
        
        if pupil:
            self.pixsize = pupil.pupil.shape[0]
            self.sizem = self.pixsize * pupil.rate
            
        if strehl:
            self.sr_factor = self.set_strehl(strehl['value'], strehl['wl'])

        self.get_filter()
        
    def set_strehl(self, sr, wl):
        factor = -np.log(sr)
        factor = sqrt(factor) * wl / self.ref
        return factor
        
    def get_strehl(self, phase, wl):
        sr = ((phase - phase.mean())**2 * (self.ref/wl)**2).mean()
        return np.exp(-sr)

    def dist(self,pixsize):
        nx = np.arange(pixsize)-pixsize/2
        gxx,gyy = np.meshgrid(nx,nx)
        freq = gxx**2 + gyy**2
        freq = np.sqrt(freq)
        return np.fft.ifftshift(freq)
        
    def get_filter(self):
        freq = self.dist(self.pixsize)/self.sizem
        freq[0,0] = 1.0
        factors = np.sqrt(0.00058)*self.r0**(-5.0/6.0)
        factors *= np.sqrt(2)*2*np.pi/self.sizem

        if not self.L0:
            self.filter = factors * freq**(-11.0/6.0)
        else:
            self.filter = factors * (freq ** 2 + self.L0**(-2))**(-11.0/12.0)

        self.filter[0,0] = 0
        
    def get_res(self):
        '''return the phase screen.
        output:
            phase screen
        '''
        if self.uncorelated:
            return self.new_phs_long_enough()
        
    def new_phs_long_enough(self):
        phase = nprand.randn(self.pixsize,self.pixsize)*np.pi
        x_phase = np.cos(phase) + 1j*np.sin(phase)
        pscreen = np.fft.ifft2(x_phase*self.filter)
        ps = np.real(pscreen)*self.pixsize**2
        
        if np.isnan(self.sr_factor):
            return ps
        else:
            return ps/ps.std()*self.sr_factor 


class SkyBG(DataFile):
    '''Sky background.
    
    '''
    def __init__(self, vapor, airmass, flag = 'mk'):
        '''
        input:
            (float) water vapor: can only be from 1.0mm, 1.6mm, 3.0mm, 5.0mm for Mauna Kea
                                and 2.3mm, 4.3mm, 76mm, 100mm for cp
            (float) airmass: 1.0, 1.5, 2.0
        keyword:
            (string) flag: 
        '''
        self.filename = "%s_skybg_zm_%d_%d_ph.dat"%(flag, int(vapor*10), int(airmass*10))
        if flag == 'mk':
            self.default_name = "mk_skybg_zm_10_10_ph.datt"
        else:
            self.default_name = "cp_skybg_zm_23_10_ph.dat"
        self.wl_factor = 1e9
        self.get_data()
        self.res_factor = 1
        
class SkyTrans(DataFile):
    def __init__(self, vapor, airmass, flag = 'mk'):
        self.filename = "%strans_zm_%d_%d.dat"%(flag, int(vapor*10), int(airmass*10))
        if flag == 'mk':
            self.default_name = "mktrans_zm_10_10.dat"
        else:
            self.default_name = "cptrans_zm_23_10.dat"
        self.wl_factor = 1e6
        self.get_data()
        self.res_factor = 1  


class Dispersion(DataFile):
    def __init__(self, RH, T, P, air_mass):
        self.filename = "/dispersion/atmo_refract_{}_{}_{}.dat".format(RH, T, P)
        self.default_name = "/dispersion/atmo_refract_10_-10_700.dat"
        self.wl_factor = 1e6
        self.out = [2]
        self.get_data()
        self.wl_eff = 3.8e-6
        self.res_factor = 1
        self.air_mass = air_mass
        
    def average_interval(self, wavelength, datax, datay, dout = None):
        return super().average_interval(wavelength, datax, datay, interp = True, dout = dout)

        
    def get_res(self, wl):
        n_real = super().get_res(wl, out_col = [2]) + 1
        ind = np.argmin(abs((wl - self.wl_eff)))
        dis = 206265 * (n_real ** 2 - 1)/(2 * n_real ** 2)
        dis -= dis[ind]
        dis = dis * np.tan(np.arccos(1/self.air_mass))
        return dis
        

class InstTransEm(object):
    def __init__(self, mirrors, mirror_temp):
        self.mirrors = mirrors
        self.temp = mirror_temp
        
        
    def blackbody(self, Temp, wl):
        hconst = 6.62607004e-34 #J/s
        kconst = 1.3806488e-23 #J/K
        splight = 3e8
        SBconst = 5.670373e-8
        totalL = SBconst * Temp ** 4
        core = np.exp(hconst * splight / wl / kconst / Temp) - 1
        res = 2 * splight  / wl ** 4 / core
        return res
    
    def get_trans(self, wavelen):
        trans = 1
        for mirror in self.mirrors:
            trans *= mirror
        return wavelen*0 + trans
    
    def get_em(self, wavelen):
        key = 0
        for mirror, temp, in zip(self.mirrors, self.temp):
            if temp:
                Oi = (1.0 - mirror) * self.blackbody(temp, wavelen)
            else:
                Oi = 0

            if key:
                Oall = (Oall * mirror + Oi)
            else:
                key = 1
                Oall = Oi
        return Oall / 206265**2 /1e9

#==================classes related with telescope========================
class Pupil(object):
    def __init__(self):
        self.pixsize = 0
        self.d = 10
        self.rate = 2.5e-3
        self.pupil = None
        self.area = None
        
    @property
    def rate(self):
        return self._rate
        
    @rate.setter
    def rate(self, _rate):
        self._rate = _rate
        self.pixsize = int(self.d * 1.1 /self._rate)
            
    def make_pupil(self):
        print('make_pupil not works for Pupil object')
        
    def load_from(self, file):
        img, header = fits.getdata(file,header=True)
        self.pixsize = img.shape[0]
        self.rate = header['rate']
        self.pupil = img
        self.d = header['diameter']
        self.type = header['type']
        
    def save_to(self, file):
        head = fits.Header()
        head['diameter'] = self.d
        head['rate'] = self.rate
        head['type'] = self.type
        fits.writeto(file, self.pupil, header = head, overwrite = True)
        
    def reduce_rate(self,rate):
        if rate > 1:
            img = self.pupil
            sz = np.array(img.shape)
            fimg = np.fft.fftshift(np.fft.fft2(img))
            new_sz = (sz/rate).astype(int)
            ori = ((sz-new_sz)/2).astype(int)

            sub_fimg = fimg[ori[0]:ori[0]+new_sz[0],ori[1]:ori[1]+new_sz[1]]
            sub_img = np.fft.ifft2(sub_fimg)
            self.rate *= float(sz[0]) / new_sz[0]
            self.pupil = abs(sub_img) / self.rate**2
        elif rate < 1:
            raise Exception('Cant imcrease rate!')
                       
    def show(self, ax = plt):
        ax.imshow(self.pupil.T, origin = 'lowerdef ')
        
    
class SqrPupil(Pupil):
    def __init__(self, d, rate):
        self.d = float(d)
        self.rate = rate
        self.type = 'square'
        self.area = self.d **2
        
    def make_pupil(self):
        pupil_sz = int(self.d / self.rate)
        total_sz = np.array([self.pixsize, self.pixsize], dtype = int)
        ori = ((total_sz - pupil_sz) / 2.).astype(int)
        self.pupil = np.zeros((total_sz[0], total_sz[1]), dtype = float)
        self.pupil[ori[0]: ori[0] + pupil_sz, ori[1]: ori[1] + pupil_sz] = 1
        self.rate = self.d * 1.0 / pupil_sz
        

class TMTPupil(Pupil):
    def __init__(self):
        # self.pixsize = 
        self.d = 30
        self.rate = 2.5E-3 # meter/pix
        
        
        self.d_seg = 1.44
        self.d_m2 = 3.9
        
        self.area = self.d_seg**2 * 1.5 * sqrt(3) / 4 * 492

        self.n_spider = 3
        self.n_line = 27
        
        self.gap = 2.5E-3
        self.width_spider = 300E-3
        self.width_spider_6 = 280E-3
        self.width_wire = 50E-3
        self.rate = 2.5E-3 # meter/pix
        
        self.pupil = []
        self.type = 'TMT'
        
    
    def _make_bar(self,alpha,point,width):
        sz = self.pupil.shape
        alp = alpha/180.*np.pi
        yy,xx = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]))
        xx = xx - (sz[0]-1.)/2 - point[0]
        yy = yy - (sz[1]-1.)/2 - point[1]
        rr = abs(-yy * np.sin(alp) + xx * np.cos(alp))
        hh = xx*np.sin(alp) + yy*np.cos(alp)
        return (rr<width/2.).astype(float) * (hh>0).astype(float)
    
    def _add_6_spider(self):
        for iang in range(6):
            ang = iang*60
            s1 = self.make_bar(ang,[0,0],self.width_spider_6/self.rate)
            self.pupil *= (1-s1) 
        
    def _add_3_spider(self):
        s1 = self.make_bar(0,[0,0],self.width_spider/self.rate)
        s2 = self.make_bar(120,[0,0],self.width_spider/self.rate)
        s3 = self.make_bar(240,[0,0],self.width_spider/self.rate)
        
        self.pupil *= (1-s1)*(1-s2)*(1-s3)
        sq3= np.sqrt(3)
        g2 = self.gap/self.rate/2*sq3
        r2 = self.d_seg /2./self.rate
        bigh = (self.n_line-1)/2.* (r2*1.5 + g2)
        bigh += r2 + g2
        
        def bar_two_point(start,end):
            width = self.width_wire/self.rate
            alpa = np.arctan2(end[0]-start[0],end[1]-start[1])
            return self.make_bar(alpa/np.pi*180,start,width)
        
        r_sec = self.d_m2/2./self.rate
        alp = 90./180.*np.pi
        xstart1 = [r_sec*np.cos(alp),r_sec*np.sin(alp)]
        alp = 210./180.*np.pi
        xstart2 = [r_sec*np.cos(alp),r_sec*np.sin(alp)]
        alp = 330./180.*np.pi
        xstart3 = [r_sec*np.cos(alp),r_sec*np.sin(alp)]
        sq3 = np.sqrt(3)
        xend1 = [bigh,bigh/sq3]
        xend2 = [-bigh,bigh/sq3]
        xend3 = [0,-bigh/np.sqrt(3)*2.]
        
        w1 = bar_two_point(xstart1,xend1)
        w2 = bar_two_point(xstart1,xend2)
        w3 = bar_two_point(xstart2,xend2)
        w4 = bar_two_point(xstart2,xend3)
        w5 = bar_two_point(xstart3,xend3)
        w6 = bar_two_point(xstart3,xend1)
        
        wires = (1-w1)*(1-w6)*(1-w2)*(1-w3)*(1-w4)*(1-w5)
        
        self.pupil *= wires
                
        
    def _add_prime(self):
        sq3 = np.sqrt(3)
        w = self.gap / self.rate
        r = self.d_seg /2. /self.rate
        h = r/2.*sq3

        Nlines = []
        Nline1=[6,11,14,15,18,19,20,21,22,23,22,23,24]
        Nlines.extend(Nline1)
        Nlines.append(23)
        Nlines.extend(Nline1[::-1])
        self.nline = len(Nlines)
        
        Nmax = np.max(Nlines)
        xr = int((Nmax+1)*(h*2+w))
        yr = xr

        cens = []
        for i,nline in enumerate(Nlines):
            yp0 = np.arange(nline)*(w+h*2)+(Nmax-nline)*(w+h*2)/2.
            xp0 = yp0*0 + (w+h*2)/2.*sq3*i
            cens.extend(zip(xp0,yp0))

        now_cen = cens[(len(cens)-1)//2]
        img_cen = (xr-1.0)/2.

        cens_x = []
        cens_y = []
        cen_region = (h * 2.2) ** 2
        for cen in cens:
            cenx = cen[0] - now_cen[0]
            ceny = cen[1] - now_cen[1]
            rcen = cenx**2 + ceny**2
            if rcen > cen_region:
                cens_x.append(cenx + img_cen)
                cens_y.append(ceny + img_cen)

        res=np.zeros((yr,xr),dtype = float)


        sub_n_x = int((r + w ) * 2) + 2
        sub_n_y = int((r + w) * sq3) + 2

        sub_yy, sub_xx = np.meshgrid(np.arange(sub_n_y), np.arange(sub_n_x))

        cen_sub_x = int(sub_n_x/2)
        cen_sub_y = int(sub_n_y/2)


        for ind,cen in enumerate(zip(cens_x,cens_y)):
            xcn = cen[0]
            ycn = cen[1]
            i_cen_x = int(xcn)
            i_cen_y = int(ycn)
            ori = [i_cen_x - cen_sub_x,i_cen_y - cen_sub_y]
            r1=abs(ori[1]+sub_yy-ycn)
            r2=abs(sq3*(ori[0]+sub_xx-xcn)-sub_yy-ori[1]+ycn)/2
            r3=abs(sq3*(xcn-sub_xx-ori[0])-sub_yy-ori[1]+ycn)/2

            rmax = np.maximum(r1,r2)
            rmax = np.maximum(rmax,r3)
            sub = (rmax <= h).astype(float)
            res[ori[0]:ori[0]+sub_n_x, ori[1]:ori[1]+sub_n_y] += sub
        self.pupil = res
        
    def _add_m2(self):
        sz = self.pupil.shape
        xx,yy = np.meshgrid(np.arange(sz[0]),np.arange(sz[1]))
        xx = xx - (sz[0]-1.)/2
        yy = yy - (sz[1]-1.)/2
        rr = xx ** 2 + yy ** 2
        rsec = self.d_m2/self.rate/2.
        sec = (rr >= rsec ** 2).astype(float)
        self.pupil *= sec
    
    def make_pupil(self):
        self._add_prime()
        self._add_m2()
        
        if self.n_spider == 3:
            self._add_3_spider()
        elif self.n_spider == 6:
            self._add_6_spider()
            
		  
class MultiBandCamera(object):
    def __init__(self, pupil):
        self.pixscal = 0.01
        self.factor = 2
        self.pixsize = 512
        self.template = None
        self.save_template = 'N'
        self.load_template = 'N' #'Y' or 'F'
        self.expt = 1
        self.pupil = pupil
        self.image = None
        self.wavelength = None
        self.save_size = None
        
    def save_to_template(self):
        head = fits.Header()
        head['phase'] = self.phase
        head['ws'] = self.wavelength[0]
        head['we'] = self.wavelength[-1]
        head['pixscal'] = self.pixscal
        
        hdulist = [fits.PrimaryHDU(header = head)]
        for img in self.image:
            hdulist.append(fits.ImageHDU(img))
            
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(self.template, overwrite = True)
        
    def load_from_template(self):
        hdulist = fits.open(self.template)
        if self.load_template == 'Y':
            header = hdulist[0].header
            assert abs(header['ws'] - self.wavelength[0]) < 1e-9, 'wavelength error'
            assert abs(header['we'] - self.wavelength[-1]) < 1e-9, 'wavelength error'
            assert len(hdulist) - 1 == len(self.wavelength), 'wavelength error'
            assert abs(header['pixscal'] - self.pixscal) < 1e-9, 'pixscal error'
            assert header['phase'] == self.phase, 'phase error'
            
        self.image = []
        for hdu in hdulist[1:]:
            self.image.append(hdu.data.astype(float))

        if self.load_template == 'F':
            ws = header['ws']
            we = header['we']
            wn = header['NAXIS3']
            self.wavelength = np.linspace(ws, we, wn)
            self.pixscal = self.pixscal
            self.phase = header['phase']
            
    def get_self_image(self, wavelength, phase = {'type':'None', 'screen':1}):
        self.wavelength = wavelength
        self.phase = phase['type']
        
        if self.load_template in ['Y', 'F']:
            self.load_from_template()
        else:
            imgs = []
            for iwl, wl in enumerate(wavelength):
                print(iwl)
                img = self.fft_img(self.pupil, wl, phase = phase['screen'])
                if self.save_size:
                    img = self.cut(img, [self.save_size, self.save_size])
                imgs.append(img)
            self.image = imgs

            if self.save_template == 'Y':
                self.save_to_template()
        

    def get_image(self, wavelength, targets = [np.array([0, 0, 1])], 
                  phase = {'type':'None', 'screen':'None'}):
        
        self.get_self_image(wavelength, phase = phase)
            
        sz = [len(wavelength), self.pixsize, self.pixsize]
        final_img = np.zeros(sz, dtype = float)
        
        for iwl,wl in enumerate(wavelength):
  
            img = self.image[iwl]
                
            for itg, pos in enumerate(targets):
                cutimg = self.cut(img, [self.pixsize, self.pixsize],
                                  center = pos[:2] / self.pixscal ) ** 2
                final_img[iwl, :, :] += cutimg/cutimg.sum() * pos[2]
            
        return final_img
        
        
    def cut2(self, psf, size, center = [0, 0]):
        '''
        cut the image to a certen size, base on the center of the image. Used in Camera Object.
        '''
        sz = psf.shape
            
        xc = center[0] + sz[0]/2
        yc = center[1] + sz[1]/2
        
        xstart = int(xc - size[0]/2)
        ystart = int(yc - size[1]/2)
        xend = xstart + size[0]
        yend = ystart + size[1]
        psf = psf[xstart:xend, ystart:yend]
        dx = (xc - size[0]/2) - xstart
        dy = (yc - size[1]/2) - ystart  
             
        return nd.interpolation.shift(psf, (dx, dy))
        
        
    def cut(self, psf, size, center = [0, 0]):
        '''
        cut the image to a certen size, base on the center of the image. Used in Camera Object.
        '''
        sz = psf.shape
        xs = 0
        ys = 0
        if sz[0]%2 != size[0]%2:
            xs = -0.5
        if sz[1]%2 != size[1]%2:
            ys = -0.5
        psf1 = nd.interpolation.shift(psf, (xs + center[0], ys + center[1]))
        psf = np.zeros(size, dtype = float)
        
        if size[0] < sz[0]:
            xstart = int((sz[0] - size[0]) / 2 + xs)
            ystart = int((sz[1] - size[1]) / 2 + ys)
            return psf1[xstart:xstart+size[0], ystart:ystart+size[1]]
        else:
            xstart = -int((sz[0] - size[0]) / 2 + xs)
            ystart = -int((sz[1] - size[1]) / 2 + ys)
            psf[xstart:xstart+sz[0], ystart:ystart+sz[1]] = psf1
            return psf
    
    def fft_img(self, pupil, wl, phase = 1):
        sz = pupil.pupil.shape
        rate = pupil.rate
        n_big = wl*180.0*3600.0/np.pi/self.pixscal/rate
        N_big = np.maximum(int(n_big), self.factor*sz[0])

        #change the N_big size for following reason:
        #1.N_big is need to be even, which is easy to be cut
        #to the pixsize of CCD which is more likely to be even
        #And also it is recommand to set the pixsize of CCD to be even
        N_big = N_big + N_big%2
        #2.searching in the interval [N_big, N_big + 18]
        #and find the value with most prime factors, which will increase the 
        #speed of FFT
        N_big_list = N_big + np.arange(10)*2
        N_big = max(N_big_list, key = primefactors)
        
        realscale = wl*180.*3600./np.pi/N_big/rate
        rate = self.pixscal/realscale
        
        bigimg = np.zeros((N_big, N_big),dtype = complex)
        
        wr = 500E-9/wl
        com = np.cos(phase*wr) + 1j*np.sin(phase*wr)
        bigimg[0:sz[0],0:sz[1]] = pupil.pupil * com
        
        fftimg = np.fft.fft2(bigimg)
        fftreal = np.fft.fftshift(abs(fftimg))
        res = reshape(fftreal[1:, 1:], rate)
        return res

        
class Lenslet(object):
    def __init__(self, arg, sub_psf_sz = 64):
        self.arg = arg
        self.ifu_n = arg['ifu_order']
        self.sub_psf_sz = sub_psf_sz
        self.sub_sz = arg['spaxel_size_px']
        self.sub_n = arg['no_spaxel']
        # spec_resoltuon may related to the prime, here I just give the normal case
        self.spec_reso = (arg['max_wavelength'] - arg['min_wavelength']) * 1e-6
        self.spec_reso /= arg['spectra_length'] - 1 # i dont know if we need to minus 1
        self.spec_length = arg['spectra_length']
        szbig = (self.sub_n + self.ifu_n * 2) * self.sub_sz + self.sub_psf_sz
        szbig = np.array([szbig, szbig]).astype(int) + 1
        self.szbig0 = arg['detector_px']
        szbig += szbig%2 + self.szbig0%2
        self.szbig = szbig
        
    def get_sub_psf(self, wavelength, rate = 32, factor = 2, pixsz = 64):
        d = self.sub_sz
        F = self.arg['lenslet_output_f']
        dia = d*1e-6*self.arg['px_pitch']
        pupil = SqrPupil(dia, dia / rate)
        pupil.make_pupil()
        phase = {'screen':pupil.pupil,'rate':pupil.rate,'type':'None'}
        ccd = MultiBandCamera(pupil)
        ccd.factor = factor        
        ccd.pixscal = 1 / F / d * 180 * 3600 / pi
        ccd.pixsize = pixsz
        psf = ccd.get_image(wavelength, phase = phase)
        fits.writeto('subpsf.fits', psf, overwrite = True)
        return psf
        
    def run_fast(self, wavelength, inpt):
        psf = self.get_sub_psf(wavelength, pixsz = self.sub_psf_sz)
        dx = 1 / sqrt(1 + self.ifu_n**2) * self.spec_length / self.sub_n
        dy = dx * self.ifu_n
        print(psf.shape, inpt.shape)
        sz = psf.shape
        szi = inpt.shape
        shifts = np.array([dx, dy, self.sub_sz,self.sub_sz])
        print(shifts)
        out_x = (int) (sz[2] + (sz[0]-1) * shifts[0] + (szi[2]-1) * shifts[2])
        out_y = (int) (sz[1] + (sz[0]-1) * shifts[1] + (szi[1]-1) * shifts[3])
        
        print(out_x, out_y)
        return ifu_patch.patch(psf, inpt, shifts )
        
    def run(self, wavelength, inpt):
        psf = self.get_sub_psf(wavelength, pixsz = self.sub_psf_sz)
        sub_s, wl_s = self.get_shifts(wavelength)
        sub_s = list(sub_s)
        print(len(sub_s))
        bigimg = np.zeros(self.szbig)
        dsz = (self.szbig - self.sub_psf_sz)/2
        ifun = self.ifu_n
        d = self.sub_psf_sz
        for iwl, wl in enumerate(wavelength):
#             if iwl > 50:
#                 break
            onepsf = psf[iwl]
            stre = inpt[iwl,:,:]
            swl = wl_s[iwl]
            print(swl)
            for ix, iy, sx, sy in sub_s:
                shifts = np.array([sx + swl, sy + swl * ifun])
                start = dsz + shifts + (d - 1)  / 2
                ints = np.round(start).astype(int)
                bigimg[ints[1]:ints[1]+d, ints[0]:ints[0]+d] += onepsf * stre[iy, ix]
#                 self.img_shift(bigimg, onepsf*stre[ix, iy], shifts)
        
        dx = (self.szbig - self.szbig0)//2
        return bigimg
#         return bigimg[dx[0]:dx[0] + self.szbig0, dx[1]:dx[1] + self.szbig0]
    
    def get_shifts(self, wavelength):
        shifts = []
        cen_xy = (self.sub_n + 1) / 2
        cen_wl = (wavelength.max() + wavelength.min())/2
        sinalp = 1 / sqrt(1 + self.ifu_n**2)
        arr_x = np.arange(self.sub_n)
        arr_y = np.arange(self.sub_n)
        dxx, dyy = np.meshgrid(arr_x, arr_y)
        sub_x = (dxx - cen_xy) * self.sub_sz
        sub_y = (dyy - cen_xy) * self.sub_sz
        shifts_wl = (wavelength - cen_wl) / self.spec_reso * sinalp
        return (zip(dxx.flatten(), dyy.flatten(), sub_x.flatten(), sub_y.flatten()), shifts_wl)
        
        
class NoiseCamera(MultiBandCamera):
    def __init__(self, pupil):
        super().__init__(pupil)
        self.readout = 10
        
        
    def get_image(self, wavelen, exp, targets, phase, s_trans, s_bkg, disp, instr, dwl):
        self.get_self_image(wavelen, phase = phase)
        sz = [len(wavelen), self.pixsize, self.pixsize]
        final_img = np.zeros(sz, dtype = float)
        bkg_img = np.zeros(sz, dtype = float)
        
        disps = disp.get_res(wavelen)
        strans = s_trans.get_res(wavelen)
        itrans = instr.get_trans(wavelen)
        sbgs = s_bkg.get_res(wavelen)
        ibgs = instr.get_em(wavelen)

        tars = []
        for target in targets:
            tars.append(target.get_res(wavelen))
            
        for iwl,wl in enumerate(wavelen):
            img = self.image[iwl]
            for itg, target in enumerate(targets):
                pos = np.array(target.pos).copy()
                pos[0] += disps[iwl]
                
                cutimg = self.cut2(img, [self.pixsize, self.pixsize],
                                  center = np.array(pos) / self.pixscal ) ** 2
                cutimg /= cutimg.sum()
                cutimg = cutimg * tars[itg][iwl] * strans[iwl] * itrans[iwl]
                
                final_img[iwl, :, :] += cutimg
                
            bkg_img[iwl, :, :] = (sbgs[iwl] * itrans[iwl] + ibgs[iwl]) * self.pixscal**2
            
        A = self.pupil.area * exp
        
        return (final_img*A*dwl, bkg_img*A*dwl)
                
                
    def make_noise(self, image):
        image = np.random.poisson(lam = image)
        image += np.random.normal(scale = self.readout, size = image.shape).astype(int)
        return image