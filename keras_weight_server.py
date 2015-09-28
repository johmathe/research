import cPickle as pickle
import zmq
import sys
import numpy

port = "5556"
if len(sys.argv) > 1:
    port = sys.argv[1]
    int(port)

context = zmq.Context()
g_socket = context.socket(zmq.REP)
g_socket.bind("tcp://*:%s" % port)
print 'server started.'

ARRAY = None
ARRAYS_RECEIVED = 0
alpha = 0.9


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

while True:
    #  Wait for next request from client
    new_array = g_socket.recv()
    print 'new array received'
    layers = pickle.loads(new_array)
    g_socket.send(pickle.dumps(layers))
    #if ARRAY is None:
    #    ARRAY = new_array
    #else:
    #    print 'computing new array...'
    #    ARRAY = alpha * ARRAY + (1-alpha) * new_array
    #ARRAYS_RECEIVED += 1
