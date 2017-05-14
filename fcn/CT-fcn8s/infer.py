import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import io

import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
im = Image.open('/home/tramac/caffe/examples/fcn/CT-fcn32s/CTData/JPEGImages/3151.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
#in_ -= np.array((166.93918,166.93918,166.93918)) # 100CT
in_ -= np.array((111.279988426,111.279988426,111.279988426)) # 512CT
in_ = in_.transpose((2,0,1))

# load net
#net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
net = caffe.Net('/home/tramac/caffe/examples/fcn/CT-fcn8s/deploy.prototxt', '/home/tramac/caffe/examples/fcn/CT-fcn8s/models/CT_512_dice_fcn8_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
#io.savemat('out.mat', {'array':out})

plt.imshow(out,cmap='gray');plt.axis('off')
plt.savefig('test.png')
plt.show()
