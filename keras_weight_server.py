import numpy as np
import cPickle as pickle
import zmq

EPOCHS = 50
PORT_BASE = 6000
N = 2


def receive_worker_data(socket):
    data = socket.recv()
    print 'received worker data'
    unpickled = pickle.loads(data)
    return unpickled


def send_worker_data(socket, z):
    socket.send(pickle.dumps(z))


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
    return np.sign(x) * np.max(0, np.abs(x) - gamma / 2)


def soft_thresholding(x, gamma):
    # TODO(johmathe): Do something nicer here
    res = []
    for l in x:
        params = []
        for p in l:
            params.append(soft_x(l, gamma))
        res.append(params)
    return res


def sum_layers(x, y):
    res = []
    for il, l in x:
        for ip, p in l:
            res.append(p + y[il][ip])
    return res


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
        u = []
        x = []
        for i in range(N):
            print 'receiving vector from worker %d' % i
            worker_data = receive_worker_data(socket[i])
            u.append(worker_data['u'])
            x.append(worker_data['x'])
            print 'received vector from worker %d' % i
        x_bar = average(x)
        u_bar = average(u)
        z = soft_thresholding(sum_layers(x_bar, u_bar))
        for i in range(N):
            # Send in mcast?
            send_worker_data(socket[i], z)
        # TODO(johmathe): stopping criterion

main_thread()
