import os
import json
import scipy.io
import pandas
import itertools
import numpy as np

from PIL import Image
from collections import OrderedDict



info = OrderedDict(description = "Testset extracted from put-in-context paper (experiment H)")

licenses = OrderedDict()

catgs = ['airplane','apple','backpack','banana','baseball bat','baseball glove','bench','bicycle','bird','boat','book','bottle','bowl','bus','cake','car','carrot','cell phone','chair','clock','cow','cup','dog','donut','fire hydrant','fork','frisbee','horse','kite','knife','motorcycle','mouse','orange','parking meter','potted plant','remote','sheep','sink','skateboard','skis','snowboard','spoon','sports ball','stop sign','suitcase','surfboard','tennis racket','tie','toothbrush','traffic light','train','truck','umbrella','vase','wine glass']


#imagedir_ori = '/home/mengmi/Projects/Proj_context2/Datasets/MSCOCO/trainColor_oriimg'
#imagedir_bin = '/home/mengmi/Projects/Proj_context2/Datasets/MSCOCO/trainColor_binimg'

imagedir_ori = '/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expH'
imagedir_bin = '/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expA'

#object_data = pandas.read_csv('/home/mengmi/Projects/Proj_context2/Datalist/trainColor_oriimg.txt', header=-1)
#binary_data = pandas.read_csv('/home/mengmi/Projects/Proj_context2/Datalist/trainColor_binimg.txt', header=-1)
#labels = pandas.read_csv('/home/mengmi/Projects/Proj_context2/Datalist/trainColor_label.txt', header=-1)


object_data = pandas.read_csv('/home/dimitar/experiments_I_and_J/expIJ/test_expJ_Color_oriimg.txt', header=-1)
binary_data = pandas.read_csv('/home/dimitar/experiments_I_and_J/expIJ/test_expJ_Color_binimg.txt', header=-1)
labels = pandas.read_csv('/home/dimitar/experiments_I_and_J/expIJ/test_expJ_Color_label.txt', header=-1)



image_cnt = 0

images = [] # fill this list with image annotations
categories = [] # fill this list with category annotations
annotations = [] # fill this list with object annotations

for (_, s), (_, s1), (_, label) in itertools.izip(object_data.iterrows(), binary_data.iterrows(), labels.iterrows()):
    
    image = Image.open(os.path.join(imagedir_ori, s[0]))
    bin_mask = np.array(Image.open(os.path.join(imagedir_bin, s1[0])))
    
    A = np.argwhere(bin_mask >= 200)
    top, left = A[0]
    bottom, right = A[-1]
    if bottom < A[-2][0] or right < A[-2][0]:
        bottom, right = A[-2]
   
    
    images.append(OrderedDict(file_name = s[0], height = image.height, width = image.width, id = image_cnt))
    annotations.append(OrderedDict(area = (bottom-top)*(right-left), iscrowd = 0, image_id = image_cnt, bbox = [left, top, right - left, bottom - top], category_id = label[0], id = image_cnt))
    image_cnt += 1


for i in range(1, 56):
    categories.append(OrderedDict(id = i, name = catgs[i-1]))




cocoannotations = OrderedDict(info = info, licenses = licenses, images = images, annotations = annotations, categories = categories)

# save annotations
with open("annotations/test_annotations_exp_J.json", "w") as f:
    json.dump(cocoannotations, f)


