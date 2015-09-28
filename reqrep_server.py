import zmq
import time
import sys
import numpy

port = "5556"
if len(sys.argv) > 1:
    port = sys.argv[1]
    int(port)

context = zmq.Context()
g_socket = context.socket(zmq.REP)
g_socket.bind("tcp://*:%s" % port)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

while True:
    #  Wait for next request from client
    print recv_array(g_socket)
    g_socket.send("World from %s" % port)
