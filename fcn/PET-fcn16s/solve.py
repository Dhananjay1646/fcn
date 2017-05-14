import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '/home/tramac/caffe/examples/fcn/PET-fcn32s/models/PET_512_fcn32_iter_100000.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
#caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/home/tramac/caffe/examples/fcn/PET-fcn32s/PETData/ImageSets/Segmentation/val.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
