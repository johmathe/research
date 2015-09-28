import threading
import keras
import cPickle as pickle
import zmq
import sys
import numpy

port = 6000

average_weights_ready = threading.Event()
layer_weights = None
n_workers = 2
sync_done = threading.Event()
data_for_worker = [threading.Event() for _ in range(n_workers)]
worker_layer = [None for _ in range(n_workers)]

def main_thread():
    global data_for_worker
    global average_weights_ready
    global average_weights
    global worker_layer
    global n_workers
    while True:
        average_weights_ready.clear()
        for worker in data_for_worker:
            worker.wait()
        average_weights = worker_layer[0]
        for i, w_layer in enumerate(worker_layer[1:]):
            for j, layer in enumerate(w_layer):
                for k, inside_layer in enumerate(layer):
                    average_weights[j][k] += worker_layer[i][j][k]
        for j, layer in enumerate(average_weights):
            for k, inside_layer in enumerate(layer):
                average_weights[j][k] /= n_workers
        average_weights_ready.set()
        print 'one pass'


def worker(i):
    global data_for_worker
    global average_weights_ready
    global average_weights
    global worker_layer
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    server_string = "tcp://*:%d" % (port + i)
    print 'thread %d listening on %s' % (i, server_string)
    socket.bind(server_string)
    while True:
        #  Wait for next request from client
        data_for_worker[i].clear()
        data = socket.recv()
        layers = pickle.loads(data)
        worker_layer[i] = layers
        data_for_worker[i].set()
        average_weights_ready.wait()
        socket.send(pickle.dumps(average_weights))

threads = []
for i in range(n_workers):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()
main_t = threading.Thread(target=main_thread)
main_t.start()
[t.join() for t in threads]
