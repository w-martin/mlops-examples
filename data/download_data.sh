#!/bin/bash
echo "Downloading iris data..."
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data &
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names &
wait
echo "Downloaded data"

