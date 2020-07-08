#%%
import ctypes
import struct
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def read_c_array(file):
    with open(file, 'rb') as fid:
        dim, = struct.unpack('i', fid.read(4))
        shape = struct.unpack('%di'%dim, fid.read(dim*4))
        size, = struct.unpack('i', fid.read(4))
        tp, = struct.unpack('c', fid.read(1))
        print('dim', dim)
        print('shape', shape)
        print('size', size)
        print('tp', tp)
        n = 1
        for i in range(dim):
            n *= shape[i]
        fmt = '%d%s'%(n, str(tp, encoding="utf-8"))
        data = struct.unpack(fmt, fid.read(n * 4))

    return np.array(data).reshape(shape)

#%%
ll = ctypes.cdll.LoadLibrary

lib = ll("../lib/libtestcu.so")
print(lib.foo(10))

print(lib.set_global(20))
print(lib.get_global())
print(lib.set_global(30))
print(lib.get_global())


A = np.arange(10, 20)
print((A**2).sum())

#%%


#%%
# res = read_c_array("../../testdata/output.bin")


img = res[0,:,:]
img[1, 1] = 1
img[1, 98] = 2000
img[98, 1] = 4000
img[98, 98] = 6000
ffts = np.fft.fftshift(img)

f, subs = plt.subplots(1, 2, figsize = (8, 4))

nf = 100
ns = 9
n1 = ns//2
ss = np.zeros((ns, ns))
ss[0:n1, 0:n1] = img[(nf - n1) : nf, (nf - n1) : nf]
ss[0:n1, n1:ns] = img[(nf - n1) : nf, 0:(ns - n1)]
ss[n1:ns, n1:ns] = img[0:(ns - n1), 0:(ns - n1)]
ss[n1:ns, 0:n1] = img[0:(ns - n1), (nf - n1) : nf]
subs[0].imshow(ss)
subs[1].imshow(ffts[46:55, 46:55])

nz = 4
big_index = np.zeros((166, 166, nz), dtype = int) - 1
subs = read_c_array("../../testdata/wfs_subs.bin")
subb = (subs - 4)/10
subb = subb.astype(int)

for i in range(res.shape[0]):
    for x in range(10):
        for y in range(10):
            if x < n1:
                xi = nf - n1 + x
            else:
                xi = x - n1

            if y < n1:
                yi = nf - n1 + y
            else:
                yi = y - n1
            bxi = subb[i, 0] * 5 + x
            byi = subb[i, 1] * 5 + y
            # print(subb[i, 0], byi, y, subb[i, 1], bxi, x)
            for bzi in range(nz):
                if big_index[byi, bxi, bzi] < 0:
                    big_index[byi, bxi, bzi] = i * 100 * 100 + yi * 100 + xi
                    break

#%%
out = res.flatten()
big_image = np.zeros((165, 165))

for i in range(165):
    for j in range(165):
        for k in range(nz):
            index = big_index[i, j, k]
            if index > 0:
                big_image[i, j] += out[index]

plt.imshow(big_image)

#%%
x = np.zeros([4, 4])
x[1, 1] = []


#%%
def cuda_patch_plan2(self):
    n_fft = self.n_big
    n_sfft = self.n_big // 3
    fft_index = np.arange(n_fft * n_fft * n_sub, dtype=int)
    fft_index.reshape((n_sub, n_fft, n_fft))

    for i in range(nsub):
        fft_index[i, :, :] = np.fft.fftshift(fft_index[i, :, :])
    
    fft_index.reshape((n_sfft, n_bin, n_sfft, n_bin))
    fft_index = np.swapaxes(fft_index, 1, 2)
    fft_index.reshape((n_sfft, n_sfft, n_bin * n_bin))

    n = self.plate_pix // self.plate_interval + 1

    for i, ap in enumerate(self.apertures):
        pos = (np.array(ap.position) / self.aperture_size) * self.plate_interval + (bigsz - self.plate_interval)/2 - edge
        pos = pos.astype(int)

        for y in range(self.plate_pix):
            for x in range(self.plate_pix):
                b = big_index.get((y + pos[1], x + pos[0]), [])
                for z in fft_index[y, x, :]:
                    b.append(z)
                big_index[(y + pos[1], x + pos[0])] = b

    ni_max = 0
    for item in big_index.items:
        ni = len(item)
        if ni > ni_max:
            ni_max = ni

    a_big_index = np.array((bigsz, bigsz, ni_max))

    for yi, xi in big_index.keys():
        d = big_index[(yi, xi)]
        ni = len(d)
        a_big_index[yi, xi, 0: ni] = d




#%%
def cuda_patch_plan1(self):
    edge = (self.plate_pix - self.plate_interval)//2
    bigsz = self.plate_interval * self.n_grid + edge * 2

    bigimg = np.zeros((bigsz, bigsz)) - 1
    big_index = {}


    n_fft = self.n_big
    n_pp = self.plate_pix
    n_c = n_pp//2
    n_intv = self.plate_interval


    for i, ap in enumerate(self.apertures):
        pos = (np.array(ap.position) / self.aperture_size) * self.plate_interval + (bigsz - self.plate_interval)/2 - edge
        pos = pos.astype(int)

        for x in range(n_pp):
            for y in range(n_pp):
                if x < n_c:
                    xi = n_fft - n_c + x
                else:
                    xi = x - n_c

                if y < n_c:
                    yi = n_fft - n_c + y
                else:
                    yi = y - n_c

                bxi = pos[0] * n_intv + x
                byi = pos[1] * n_intv + y

                d = big_index.get((byi, bxi), [])
                d.append(i * n_fft* n_fft + (yi+ by) * n_fft + xi + bx)

                big_index[(byi, bxi)] = d

    ni_max = 0
    for item in big_index.items:
        ni = len(item)
        if ni > ni_max:
            ni_max = ni

    a_big_index = np.array((bigsz, bigsz, ni_max))

    for yi, xi in big_index.keys():
        d = big_index[(yi, xi)]
        ni = len(d)
        a_big_index[yi, xi, 0: ni] = d

    return bigimg



#%%
from astropy.io import fits
res = read_c_array("../../testdata/wfs_out.bin")
plt.imshow(res)
fits.writeto("cres.fits", res, overwrite=True)
#%%
res2 = read_c_array("../../testdata/wfs_image.bin")
plt.imshow(res2)
fits.writeto("pres.fits", res2, overwrite=True)

# %%

img = fits.getdata("../milestone.fits")
plt.imshow(img)


# %%

data = read_c_array('../endend.bin')

print(data[10, 1])
plt.imshow(data)


# %%

middle = read_c_array('../middle.bin')

debug = fits.getdata('../debug_middle.fits')

sp = middle.shape
md = np.zeros((sp[0], sp[1]*2, sp[2]))
md[:,0:sp[1],:] = middle
md[:,sp[1]:,:] = debug

fits.writeto('../middles.fits', md, overwrite=True)
f, subs = plt.subplots(1, 2, figsize = (10, 5))

i = 3
subs[0].imshow(middle[i,:,:])
subs[1].imshow(debug[i,:,:])
# %%

psf = read_c_array('../middle.bin')
print(psf.shape)
plt.imshow(psf[0,:,:])


# %%
psf = fits.getdata('../cuwfs.fits')
plt.imshow(psf)


# %%
