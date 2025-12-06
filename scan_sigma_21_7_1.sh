# export OMP_NUM_THREADS=64
# export MKL_NUM_THREADS=64

echo "Scanning m=3..."
for s in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
do
    echo " -> Running sigma=$s"
    uv run main.py --n_sites 21 --n_fermions 7 --sigma $s --num_evecs 30 --save_name "solenoid_21_7_fine"
done

echo "All done!"