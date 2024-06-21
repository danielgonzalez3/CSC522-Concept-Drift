#!/bin/bash

# pip install river==0.9.0

# for i in {1..5}
# do
#    python3 serial_batched.py
# done

# pip install river==0.21.0

for i in {1..5}
do
   python3 parallel_online.py
done
