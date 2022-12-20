import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import scipy
from PIL import Image
from scipy.fft import fft, ifft, fft2, ifft2
import argparse

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='image')
    # parser.add_argument('--no-plot', type=str, help = 'prevents plotting')
    args = parser.parse_args()
    return args.name

def checkPip():
    versions = f'\
    numpy version : {np.__version__}\n\
    pandas version : {pd.__version__}\n\
    matplotlib version : {plt.__version__}\n\
    scipy version : {scipy.__version__}\n\
    PIL version : {PIL.__version__}\
    '
    return versions

def createImage(path):
    imageFormat = Image.open(path)
    imageArray = np.asarray(imageFormat)
    imageArrayFlat = imageArray.mean(axis=-1)

    d = {'raw image':imageFormat, 'array image':imageArrayFlat}
    return d

def gaussian(x, y, stdx, stdy, x0, y0):
    return np.exp(-((x-x0)**2/(2*stdx**2)+(y -y0)**2/(2*stdy**2)))



def fourierTransform(image):
    img = image.get('array image')
    imgFT = fft2(img)
    FTx = np.fft.fftfreq(img.shape[0], d=10)
    FTy = np.fft.fftfreq(img.shape[1], d=10)
    d = {"xFT":np.abs(imgFT).sum(0), "yFT":np.abs(imgFT).sum(1), "freqX":FTx, "freqY":FTy}
    return d

def plotFT(image, fourierTransform, plot = True):
    
    

    h,w = image.get("array image").shape
    x,y = np.arange(h), np.arange(w)
    
    beamx, beamy = np.meshgrid(np.linspace(-h/2, h/2, h),np.linspace(-w/2, w/2, w))
    

    xFT = fourierTransform.get("xFT")
    yFT = fourierTransform.get("yFT")
    normxFT = xFT/np.max(xFT)
    normyFT = yFT/np.max(yFT)
    
    xdata = image.get("array image").sum(0)
    ydata = image.get("array image").sum(1)
    # print(xdata.argmax(), ydata.argmax())
    normxdata = xdata/np.max(xdata)
    normydata = ydata/np.max(ydata)
    
    nl = "\n"
    # for i, x in enumerate(normxdata):
    # print(f"x : {type(np.where((normxdata>0.500) & (normxdata<0.501)))}, \nmax : {np.argmax(normxdata)}")
    m = normxdata.argmax()
    fwhm = np.where((normxdata>0.500) & (normxdata<0.501))
    l, u = fwhm[0][0], fwhm[0][1]
    # max at index 642, half max at index 
    # print(f"max index : {m}")
    print(f"half max index : {l}{nl}{u}{nl}distance from max to lower halfmax : {m-(u-l)}")
    beam = gaussian(beamx, beamy, 180 , 180, 0, 0)
    print(x)
    print(y)
    if plot is not False:
        _, [[Y, im],[_, X]] = plt.subplots(2,2, figsize = (12,12),gridspec_kw = {'height_ratios':[2,1], 'width_ratios' : [1,2]})
        # im.imshow(image.get('array image'))
        im.imshow(beam, cmap="gray")
        im.set_aspect('auto', share=True)
        im.grid()
        im.set_xticklabels([]),im.set_yticklabels([])
        
        

        X.plot(x, normxdata, label = 'x plot')
        X.plot(x, np.fft.fftshift(normxFT), label = "x FT")
        X.plot(x, beam.sum(0)/np.max(beam.sum(0)), label = "generated gaussian")
        X.legend()
        X.grid()

        Y.plot(normydata, y, label = 'y plot')
        Y.plot(np.fft.fftshift(normyFT), y, label = "y FT")
        Y.legend()
        Y.grid()

        _.axis('off')
        plt.savefig('plot.png', dpi = 300)
        # plt.show()


def make_gaussian(size, sigma):
    x, y = np.linspace(-size/2, size/2, size), np.linspace(-size/2, size/2, size)
    x, y = np.meshgrid(x,y)

    z = np.exp(-((x**2)+(y**2))/(2*(sigma**2)))
    return z

def plot_fourier_transform(image):
    image = np.sum(image, axis=1)
    FT = np.fft.fft(image)
    FT_shift = np.fft.fftshift(FT)
    FT_mag = np.abs(200*FT_shift)
    # FT_mag = np.log10(FT_mag)
    FT_mag = FT_mag/np.max(FT_mag)
    plt.plot(image/max(image))
    plt.plot(FT_mag)


size = 128
sigma = 10
image = make_gaussian(size, sigma)

shiftx = 30
shifty = 30
image_shifted = make_gaussian(size,sigma)
image_shifted[shifty:, shiftx:] = image[:size-shifty,:size-shiftx]

plt.subplot(2,2,1)
plt.imshow(image, cmap="gray")
plt.title("original")
plt.subplot(2,2,2)
plot_fourier_transform(image)
plt.title("1D fourier transform")
plt.subplot(2,2,3)
plt.imshow(image_shifted, cmap="gray")
plt.title("shifted")
plt.subplot(2,2,4)
plot_fourier_transform(image_shifted)
plt.title("1d shifted fourier tranform")
plt.show()

def saveToFrame():
    pass

def main():
    pass
    # file = arg()
    # im = createImage(file)
    # FTim = fourierTransform(im)
    # print(im.get('array image'))
    # plotFT(im, FTim, plot=True)
    

if __name__ == '__main__':
    main()


# import numpy as np
# import matplotlib.pyplot as plt

# L=1
# N=2048

# #print(shape)
# x = np.arange(-L,L,2.*L/N)

# y= np.arange(-L,L,2.*L/N)

# xx, yy=np.meshgrid(x,y)
# #print(xx,yy)
# #print(np.size(x))
# kx= np.fft.fftfreq(np.size(x),2.*L/N)
# ky=np.fft.fftfreq(np.size(y),2.*L/N)
# kxx,kyy= np.fft.fftshift(np.meshgrid(kx,ky))
# #print(xx)

# gauss= np.exp(-xx**2+yy**2)
# #print(gauss)

# plt.figure()
# plt.pcolormesh(xx,yy, gauss)
# plt.gca().set_aspect("equal") 
# plt.colorbar()
# # plt.show()


# FFT=np.fft.fft2(gauss)
# FFT_Shift = np.fft.fftshift(FFT)
# FFT_Log= np.log(FFT)
# #print(FFT)

# # plt.figure()
# # This one does not work
# # plt.pcolormesh(kxx,kyy,abs(np.fft.fftshift(FFT_Log)))
# plt.plot(kxx.sum(0), np.fft.fftshift(FFT_Log.sum(0)))
# # However, using imshow works fine
# # plt.imshow(abs(FFT_Log)) # This one works
# plt.gca().set_aspect("equal") 
# plt.colorbar()
# plt.show()


"""
arim = image.get('array image')
    height, width  = arim.shape
    x,y = np.arange(start = np.floor(-height/2), stop = np.floor(height/2)), np.arange(start = np.floor(-width/2), stop = np.floor(width/2))
    
    
    xdata = arim.sum(0)
    ydata = arim.sum(1)
    maxX = np.max(xdata)#np.max(image.sum(0))
    maxY = np.max(ydata)#np.max(image.sum(1))

    imageFT2 = np.fft.fftshift(fft2(np.fft.ifftshift(image.get('array image'))))

    ximageFT2 = imageFT2.sum(0)
    yimageFT2 = imageFT2.sum(1)
    maxXFT2 = np.max(ximageFT2)
    maxYFT2 = np.max(yimageFT2)

    xdataFT = np.abs(np.fft.fft(ydata))
    ydataFT = np.abs(np.fft.fft(ydata))
    maxXFT = np.max(xdataFT)
    maxYFT = np.max(ydataFT)

    xmu = np.mean(xdata/maxX)
    xsigma = np.std(xdata/maxX)
    ymu = np.mean(ydata/maxY)
    ysigma = np.std(ydata/maxY)


    T = 1/800
    xf = np.linspace(0, 1/(2*T), len(x))
"""