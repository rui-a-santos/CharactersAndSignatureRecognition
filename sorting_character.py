import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops

class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename,1)

        #apply threshold in order to make the image binary
        bw = image < 210

        # remove artifacts connected to image border
        cleared = bw.copy()

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1

        fig = plt.figure()

        letters = list()
        order = list()

        regions = 0
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            # skip small images
            if maxr - minr > len(image)/130:
            #if maxr - minr > len(image)/104: # better to use height rather than area.
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)
                regions += 1

        print("Regions = ", regions)
        #sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        #worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])

        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)

        for x in range(len(lines)):
            lines[x].sort(key=lambda tup: tup[1])

        final = list()

        #Declares lowest point in a row and the current row youre in
        lowestPoint = 0
        thisRow = 0

        #Loops through all the regions, gets the lowest point in a given line. This point is used to determine the lines. You only do this if there is a line below.
        for x in range(len(order)):
            #Determines if there is a lower point in the line
            if order[x][0] > lowestPoint:
                #As the order is higher than the lowest point, then there is a new lowest point in line so we need to sort the line. You sort by key i, which is minc
                order[thisRow:x] = sorted(order[thisRow:x], key=lambda i: i[1])
                #Current row needs to be changed to x
                thisRow = x
            #Changes the lowest point to the new lowest point in the line
            lowestPoint = max(lowestPoint,order[x][2])

        #If there isnt a line below, it sorts this line as it is the last row
        order[thisRow:] = sorted(order[thisRow:], key=lambda i: i[1])

        #Gets the region out of the image and then appends it to final
        for x in order:
            minr, minc, maxr, maxc = x
            region = bw[minr:maxr, minc:maxc]
            region = imresize(region.astype('float32'), (28,28))
            final.append(region)
        return final

    def __init__(self):
        print ("Extracting characters...")
