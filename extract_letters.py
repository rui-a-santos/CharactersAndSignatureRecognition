import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops

image = imread('./ocr/training/training4.png',1)

#apply threshold in order to make the image binary
bw = image < 210

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared,neighbors=8)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(bw, cmap='jet')

count=0

for region in regionprops(label_image):
    minr, minc, maxr, maxc = region.bbox
    # skip small images
    if maxr - minr > len(image)/130:

        # draw rectangle around segmented coins
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        count+=1

plt.show()
print(count)
