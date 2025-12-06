import subprocess
import numpy as np

# echo "Scanning m=3..."
# for s in 0.025 0.075 0.125 0.175 0.225 0.275 0.325 0.375 0.425 0.475 0.525 0.575 0.625 0.675 0.725 0.775 0.825 0.875 0.925 0.975
# do
#     echo " -> Running sigma=$s"
#     uv run main.py --n_sites 21 --n_fermions 7 --sigma $s --num_evecs 30 --save_name "solenoid_21_7_fine"
# done

# echo "All done!"


print("Scanning m=3...")

# Generate the range of sigma values: 0.025 to 0.975 with step 0.05
sigmas = np.arange(0.025, 2.01, 0.025)

for s in sigmas:
    # Format s to ensure it looks like the bash output (though python floats are fine)
    s_val = f"{s:.3f}"
    print(f" -> Running sigma={s_val}")
    
    command = [
        "uv", "run", "main.py",
        "--n_sites", "21",
        "--n_fermions", "7",
        "--sigma", s_val,
        "--num_evecs", "30",
        "--save_name", "solenoid_21_7"
    ]
    
    subprocess.run(command, check=True)

print("All done!")