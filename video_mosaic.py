import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import re

class Mosaic():
    def __init__(self, frame_num):
        self.frame_num = frame_num
        
    def calc_l2_dist(self, a, b, c, x, y, z):
        return (abs(int(a)-int(x)) + abs(b-y) + abs(c-z))

    def find_nearest_img(self, r, g, b, data):
        least_dist = np.inf
        least_row = None

        for row in data.values:
            dist = self.calc_l2_dist(r, g, b, row[1], row[2], row[3])
            
            if dist < least_dist:
                least_dist = dist
                least_row = row

        file = least_row[0]
   
        # remove dupes
        data = data.drop(data.index[data['Filename'] == file].tolist()) # cpu

        return file, data

    def construct_mosaic(self, pixel_batch, r, g, b, data):
        # create empty matrix of target image size
        mosaic = np.zeros((r.shape[0], r.shape[1], 3))

        # iterate through RGB and find the closest related image
        for x in range(0, r.shape[0], pixel_batch):
            for y in range(0, r.shape[1], pixel_batch):

                # need to handle padding issues later
                r_batch = r[x:x+pixel_batch, y:y+pixel_batch]
                g_batch = b[x:x+pixel_batch, y:y+pixel_batch]
                b_batch = g[x:x+pixel_batch, y:y+pixel_batch]

                # computer batch averages
                r_avg = np.mean(r_batch)
                g_avg = np.mean(g_batch)
                b_avg = np.mean(b_batch)

                # find the closest image
                # returns the filename
                img,data = self.find_nearest_img(r_avg, b_avg, g_avg, data)
                tmp_img_name = img

                # extract the filename
                img = Image.open('data/images/'+img)
                tmp_img = img

                # resize the image to be the batch size
                img = img.resize((pixel_batch, pixel_batch))#, Image.ANTIALIAS)

                # Convert to opencv image
                img = np.array(img) 
                img = img[:, :, ::-1].copy() 

                # add the image to the mosaic
                mosaic[x:x+pixel_batch, y:y+pixel_batch, 0] = img[:,:,2]
                mosaic[x:x+pixel_batch, y:y+pixel_batch, 1] = img[:,:,1]
                mosaic[x:x+pixel_batch, y:y+pixel_batch, 2] = img[:,:,0]

        out = Image.fromarray(mosaic.astype(np.uint8))
        return out

    #read in the target image and divide it into RGB channels
    def extract_rgb(self, target):
        image = cv2.imread(target)
        b,g,r = cv2.split(image)

        return[r, g, b]

    def run(self):
        print ("Mosaicing frame " + str(self.frame_num))
        pixel_batch = 10
        video_frames = 'data/videos/bunny_frames/'
        rgb_data = pd.read_csv('data/avg_database.csv')

        # recopy the saved rgb avg data because images are removed from the copy when used
        data = rgb_data.copy()

        # the path to the frame
        file =  video_frames + 'frame'+str(self.frame_num)+'.jpg'

        # extract RGB from the frame
        r,g,b = self.extract_rgb(file)

        print('constructing mosaic...')
        # construct a mosaic for the frame
        image = self.construct_mosaic(pixel_batch, r, g, b, data)
        print('done constructing mosaic...')

        # retrieve the frame number from the file name and
        # save the mosaic frame with the same frame number
        image.save('data/videos/tmp/frame%d.jpg' % int(self.frame_num))


if __name__=="__main__":
    # frames = [0,1,2,4,5,6,7,10]
    frames = [11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,30]
    # frames = [21,22,23,24,25,26,27,30]
    # frames = [31,32,33,34,35,36,37,38,39,40]
    mosaics = []
   
    for frame in frames:
        mosaics.append(Mosaic(frame))
        mosaics[-1].run()
        