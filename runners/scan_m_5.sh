echo "Scanning m=5..."
for s in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo " -> Running sigma=$s"
    uv run main.py --n_sites 25 --n_fermions 5 --sigma $s --save_name "solenoid_m_5"
done

echo "All done!"