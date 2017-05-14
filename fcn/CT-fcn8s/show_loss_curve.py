import os
import random
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks

train_log_file = open('/home/tramac/caffe/examples/fcn/CT-fcn8s/log') # The log file.
train_interval = 20        # display value in solver.prototxt.
test_interval = 736        # test_interval value in solver.prototxt.

string_output = os.popen("cat train_log_file | grep 'mean accuracy' | awk '{print $8}'").readlines()
train_loss = string_output
idx_train = [(x - 1) * train_interval for x in range(1, len(train_loss) + 1)]
string_output = os.popen("cat train_log_file | grep 'mean accuracy' | awk '{print $8}'").readlines()
test_loss = string_output
idx_test = [(x - 1) * test_interval for x in range(1, len(test_loss) + 1)]

color = [random.random(), random.random(), random.random()]
linewidth = 0.75

plt.plot(idx_train, train_loss, color = color, linewidth = linewidth)
plt.plot(idx_test, test_loss, color = color, linewidth = linewidth)
plt.legend('Train Loss', 'Test Loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('Train & Test Loss Curve')
plt.savefig(Curve.png)
plt.show()