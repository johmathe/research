import numpy as np
import cPickle as pickle
import zmq
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCHS = 200
PORT_BASE = 6000
N = 2


def receive_worker_data(socket):
    data = socket.recv()
    unpickled = pickle.loads(data)
    return unpickled


def send_worker_data(socket, x_bar):
    socket.send(pickle.dumps(x_bar))


def average(worker_data):
    # TODO(johmathe): Do something nicer here

    n = len(worker_data)
    res = [[p / n for p in l] for l in worker_data[0]]
    for worker in worker_data[1:]:
        for il, l in enumerate(worker):
            for ip, p in enumerate(l):
                res[il][ip] += p / n
    assert len(res) == len(worker_data[0])
    assert len(res[0]) == len(worker_data[0][0])
    return res


def soft_x(x, gamma):
    return np.sign(x) * np.maximum(0, np.abs(x) - gamma / 2)


def soft_thresholding(x, gamma):
    # TODO(johmathe): Do something nicer here
    res = []
    for l in x:
        params = []
        for p in l:
            params.append(soft_x(p, gamma))
        res.append(params)
    return res


def sum_layers(x, y):
    res = []
    for il, l in enumerate(x):
        params = []
        for ip, p in enumerate(l):
            params.append(p + y[il][ip])
        res.append(params)
    return res

def plot_all(workers_x, x_bar, i):
    prefix = './results/epoch_%d/' % i
    os.mkdir(prefix)
    for iw, w in enumerate(workers_x):
        for il, l in enumerate(w):
            for ip, p in enumerate(l):
                plt.figure()
                plt.hist(p.flatten(), bins=np.linspace(-1, 1, 70))
                plt.savefig('%s/w%d_l%d_p%d.png' % (prefix, iw, il, ip))
    for il, l in enumerate(x_bar):
        for ip, p in enumerate(l):
            plt.figure()
            plt.hist(p.flatten(), bins=np.linspace(-1, 1, 70))
            plt.savefig('%s/x_bar_l%d_p%d.png' % (prefix, il, ip))

def epsilon(x, y):
    total = 0
    for il, l in enumerate(x):
        for ip, p in enumerate(l):
            total += np.sum((p - y[il][ip]) ** 2)
    return np.sqrt(total)

def main_thread():
    socket = []
    for i in range(N):
        port = PORT_BASE + i
        context = zmq.Context()
        socket.append(context.socket(zmq.REP))
        address = "tcp://*:%s" % port
        print 'listening on %s' % address
        socket[i].bind(address)
    for k in range(EPOCHS):
        print 'epoch %d' % k
        # Receive all weights
        worker_data = []
        x = []
        for i in range(N):
            worker_data = receive_worker_data(socket[i])
            x.append(worker_data)
        x_bar = average(x)
        for i in range(N):
            print 'distance for v%d: %f' % (i, epsilon(x_bar, x[i]))

        #plot_all(x, x_bar, k)
        for i in range(N):
            # Send in mcast?
            send_worker_data(socket[i], x_bar)
        # TODO(johmathe): stopping criterion

main_thread()
