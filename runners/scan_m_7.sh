echo "Scanning m=7..."
for s in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo " -> Running sigma=$s"
    uv run main.py --n_sites 28 --n_fermions 4 --sigma $s --save_name "solenoid_m_7"
done

echo "All done!"