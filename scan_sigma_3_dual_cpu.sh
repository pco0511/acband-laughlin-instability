#!/bin/bash

trap "echo ' -> KeyboardInterrupt: Killing all processes...'; kill 0" SIGINT

echo "Scanning m=3..."
export PYTHONUNBUFFERED=1

(
    for s in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
    do
        echo "Running sigma=$s" | sed -u 's/^/[Node 0] /'
        taskset -c 0-31,64-95 \
        uv run main.py --n_sites 27 --n_fermions 9 --sigma $s \
           --num_evecs 20 --save_name "solenoid_m_3" --simple_out \
        2>&1 | sed -u 's/^/[Node 0] /'
    done
) &

(
    for s in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95
    do
        echo "Running sigma=$s" | sed -u 's/^/[Node 1] /'
        taskset -c 32-63,96-127 \
        uv run main.py --n_sites 27 --n_fermions 9 --sigma $s \
           --num_evecs 20 --save_name "solenoid_m_3" --simple_out \
        2>&1 | sed -u 's/^/[Node 1] /'
    done
) &

wait
echo "All done!"