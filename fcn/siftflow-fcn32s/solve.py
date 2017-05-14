import caffe
import surgery, score

import numpy as np
import os
import sys

sys.path.append('/home/tramac/caffe/examples/fcn')

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

#weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
#weights = '/home/tramac/caffe/examples/fcn/ilsvrc-nets/vgg16-fcn.caffemodel'
vgg_weights = '/home/tramac/caffe/examples/fcn/ilsvrc-nets/vgg16-fcn.caffemodel'
vgg_proto = '/home/tramac/caffe/examples/fcn/ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'
# init
#caffe.set_device(int(sys.argv[1]))  change
#caffe.set_mode_gpu()                change

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
#test = np.loadtxt('../data/sift-flow/test.txt', dtype=str)
test = np.loadtxt('/home/tramac/caffe/examples/fcn/data/sift-flow/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, False, test, layer='score_sem', gt='sem')
    score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
