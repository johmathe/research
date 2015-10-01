#!/bin/bash
killall python
sleep 1
python keras_weight_server.py &
python cifar10_cnn.py 0 localhost > /dev/null  &
python cifar10_cnn.py 1 localhost &
