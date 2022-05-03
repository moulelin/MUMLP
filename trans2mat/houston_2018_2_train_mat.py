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
houston = gdal.Open("Houston/cut/cut.dat") # Change it
#.tif  (1202, 4768) uint8
# .pix (50, 1202, 4172)

data = DatasetReadAsArray(houston)

print(data.shape)
temp_data = data.transpose(1,2,0)

print(temp_data.shape)

sio.savemat('mat/Houston_2018_train.mat', {'Houston_2018': data})

# houston = gdal.Open("Houston/cut/cut.dat") # Change it
# # (50, 601, 2384) uint16
# # (2384, 601, 50)
#
# data = DatasetReadAsArray(houston)
# driver =gdal.GetDriverByName("GTiff")
# outdata = driver.Create("new_image22.tif",xsize=2384,ysize=601,bands=50)
#
#
# outband.WriteArray(data[:,:,:].astype(float))
# outdata.FlushCache()
# outdata = None
#
# print("*"*10)
# print(data.shape)
# # data = data.transpose(1,2,0)
# # houston = data.transpose()
# # houston = houston.reshape(601,2384)
# # cv.imshow("image",data[:,:,0])
# # cv.waitKey(0)
# # cv.destroyAllWindows()
# print(data.shape)
#
# sio.savemat('mat/Houston_2018_train.mat', {'Houston_2018': data})