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

def gaussian(x, std, mean):
    y = 1/(np.sqrt(2*np.pi*std**2))*np.exp(-(x-mean**2)/(2*std**2))
    return y

def plotFT(image):
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

    xdataFT = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ydata))))
    ydataFT = np.abs(ifft(ydata))
    maxXFT = np.max(xdataFT)
    maxYFT = np.max(ydataFT)

    xmu = np.mean(xdata/maxX)
    xsigma = np.std(xdata/maxX)
    ymu = np.mean(ydata/maxY)
    ysigma = np.std(ydata/maxY)


    _, [[Y, im],[_, X]] = plt.subplots(2,2, figsize = (12,12),gridspec_kw = {'height_ratios':[2,1], 'width_ratios' : [1,2]})
    im.imshow(image.get('array image'))
    im.set_aspect('auto', share=True)
    im.grid()
    im.set_xticklabels([]),im.set_yticklabels([])
   
    print(np.abs(maxXFT2))
    
    X.plot(x, xdata/maxX, label = 'x plot')
    X.plot(x, np.abs((ximageFT2))/np.abs(maxXFT2), label = 'x Fourier transform')
    # X.set_xlim([-10,10])
    # X.plot(x, ximageFT2/maxXFT2, label = 'x 2D Fourier transform')
    # X.plot(x, np.exp(-x**2)/maxX, label = 'x Gaussian')
    X.legend()
    X.grid()

    Y.plot(ydata/maxY, y, label = 'y plot')
    # Y.plot(ydataFT/maxYFT, y , label = 'y Fourier transform')
    # Y.plot(yimageFT2/maxYFT2, y,label = 'y 2D Fourier transform')
    # Y.plot(gaussian(y, ysigma, ymu)/maxY, y, label = 'y Gaussian')
    Y.legend()
    Y.grid()

    _.axis('off')
    plt.savefig('plot.png', dpi = 100)
    # plt.show()

def saveToFrame():
    pass

def main():
    file = arg()
    im = createImage(file)
    # print(im.get('array image'))
    plotFT(im)

if __name__ == '__main__':
    main()