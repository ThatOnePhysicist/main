import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.fft import fft, ifft, fft2, ifft2

versions = f'\
numpy version : {np.__version__}\n\
pandas version : {pd.__version__}\n\
matplotlib version : {plt.__version__}\
'

print(versions)