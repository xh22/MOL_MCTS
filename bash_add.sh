#!/bin/bash

for varible1 in {1..2}
    #for varible1 in 1 2 3 4 5
do
{
    echo "Welcome $varible1 "
    python train.py to_test_add.csv $varible1
} &
done
wait
