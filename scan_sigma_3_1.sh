export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

echo "Scanning m=3..."
for s in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
do
    echo " -> Running sigma=$s"
    taskset -c 32-63,96-127 uv run main.py --n_sites 27 --n_fermions 9 --sigma $s --num_evecs 10 --save_name "solenoid_27_9"
done

echo "All done!"