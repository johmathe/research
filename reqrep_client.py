import time
import numpy as np
import zmq
import sys

address = "tcp://localhost:5556"
if len(sys.argv) > 1:
    address = sys.argv[1]

context = zmq.Context()
print "Connecting to server..."
g_socket = context.socket(zmq.REQ)
g_socket.connect(address)


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

#  Do 10 requests, waiting each time for a response
for request in range(1, 10):
    array = np.random.random(size=(100, 1024, 1024)).astype(np.float32)
    start = time.time()
    send_array(g_socket, array)
    #  Get the reply.
    message = g_socket.recv()
    print time.time() - start
    print "Received reply ", request, "[", message, "]"
