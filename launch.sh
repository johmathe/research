#!/bin/bash
killall python
sleep 1
python keras_weight_server.py &
python cifar10_cnn.py 0  &
python cifar10_cnn.py 1  &
