import cPickle as pickle
import zmq

PORT_BASE = 6000
N = 2


def receive_worker_data(socket):
    data = socket.recv()
    unpickled = pickle.loads(data)
    return unpickled['x'], unpickled['u']


def send_worker_data(socket, z):
    socket.send(pickle.dumps(z))


def average(x):
    return x


def soft_thresholding(x):
    return x


def main_thread():
    socket = []
    for i in range(N):
        port = PORT_BASE + i
        context = zmq.Context()
        socket[i] = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % port)
    while True:
        x = [None] * N
        u = [None] * N
        # Receive all weights
        for i in range(N):
            x[i], u[i] = receive_worker_data(socket[i])
        x_bar = average(x)
        u_bar = average(u)
        z = soft_thresholding(x_bar + u_bar)
        for i in range(N):
            # Send in mcast?
            send_worker_data(socket[i], z)
        # TODO(johmathe): stopping criterion

main_thread()
