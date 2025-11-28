echo "Scanning m=3..."
for s in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo " -> Running sigma=$s"
    uv run main.py --n_sites 27 --n_fermions 9 --sigma $s --save_name "solenoid_m_3"
done

echo "All done!"