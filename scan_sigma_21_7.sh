# export OMP_NUM_THREADS=64
# export MKL_NUM_THREADS=64

echo "Scanning m=3..."
for s in 0.025 0.075 0.125 0.175 0.225 0.275 0.325 0.375 0.425 0.475 0.525 0.575 0.625 0.675 0.725 0.775 0.825 0.875 0.925 0.975
do
    echo " -> Running sigma=$s"
    uv run main.py --n_sites 21 --n_fermions 7 --sigma $s --num_evecs 30 --save_name "solenoid_21_7_fine"
done

echo "All done!"