import caffe
import surgery, score

import numpy as np
import os
import sys

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

#weights = '/home/tramac/caffe/examples/fcn/ilsvrc-nets/vgg16-fcn.caffemodel'
vgg_weights = '/home/tramac/caffe/examples/fcn/ilsvrc-nets/vgg16-fcn.caffemodel'
vgg_proto = '/home/tramac/caffe/examples/fcn/ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'


# init
#caffe.set_device(int(sys.argv[1]))
#caffe.set_mode_gpu()

#solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
solver = caffe.SGDSolver('solver.prototxt')
vgg_net = caffe.Net(vgg_proto, vgg_weights, caffe.TRAIN)
surgery.transplant(solver.net, vgg_net)
del vgg_net


# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('/home/tramac/caffe/examples/fcn/data/pascal/segvalid11.txt', dtype=str)
val = np.loadtxt('/home/tramac/caffe/examples/fcn/PET-fcn32s/PETData/ImageSets/Segmentation/val.txt', dtype=str)
for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
