# -*- coding: utf-8 -*-
'''
@File  : houston_2018_2_label_mat.py
@Author: Moule Lin
@Date  : 2021/6/23 19:24
@Github: https://github.com/moulelin
'''
from osgeo.gdal_array import DatasetReadAsArray
from osgeo import gdal
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2 as cv
houston = gdal.Open("Houston/label/label_resample.tif") # Change it
# (50, 601, 2384) uint16
# (2384, 601, 50)

data = DatasetReadAsArray(houston)
driver =gdal.GetDriverByName("GTiff")
outdata = driver.Create("new_image22.tif",xsize=2384,ysize=601,bands=1)
outband = outdata.GetRasterBand(1)
outband.WriteArray(data[:,:].astype(int))
outdata.FlushCache()
outdata = None

print("*"*10)
print(data.shape)
# data = data.transpose(1,2,0)
# houston = data.transpose()
# houston = houston.reshape(601,2384)
# cv.imshow("image",data[:,:,0])
# cv.waitKey(0)
# cv.destroyAllWindows()
print(data.shape)

sio.savemat('mat/Houston_2018_label.mat', {'Houston_2018': data})