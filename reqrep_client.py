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

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


#  Do 10 requests, waiting each time for a response
for request in range(1, 10):
    array = request * np.ones((100, 1024, 1024)).astype(np.float32)
    start = time.time()
    print 'sending new array'
    print 'MEEH: %s' % send_array(g_socket, array)
    #  Get the reply.
    message = recv_array(g_socket)
    print time.time() - start
    print "Received reply ", request, "[", message, "]"
