export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

echo "Scanning m=3..."
for s in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95
do
    echo " -> Running sigma=$s"
    taskset -c 0-31,64-95 uv run main.py --n_sites 27 --n_fermions 9 --sigma $s --num_evecs 10 --save_name "solenoid_27_9"
done

echo "All done!"