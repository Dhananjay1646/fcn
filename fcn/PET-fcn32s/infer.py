import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
im = Image.open('/home/tramac/caffe/examples/fcn/PET-fcn32s/PETData/JPEGImages/2905.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((11.5746805647,11.5746805647,11.5746805647))
in_ = in_.transpose((2,0,1))

# load net
#net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
net = caffe.Net('/home/tramac/caffe/examples/fcn/PET-fcn32s/deploy.prototxt', '/home/tramac/caffe/examples/fcn/PET-fcn32s/models/PET_512_dice_fcn32_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

plt.imshow(out,cmap='gray');plt.axis('off')
plt.savefig('test.png')
plt.show()
