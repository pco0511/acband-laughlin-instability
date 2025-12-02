echo "Scanning m=3..."

# Python이 출력을 버퍼링하지 않고 즉시 뱉도록 설정 (실시간 로그 확인용)
export PYTHONUNBUFFERED=1

(
    for s in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
    do
        # [Node 0] Running... 메시지 자체도 prefix를 붙여서 출력
        echo "Running sigma=$s" | sed -u 's/^/[Node 0] /'
        
        numactl --cpunodebind=0 --membind=0 \
        uv run main.py --n_sites 27 --n_fermions 9 --sigma $s \
           --num_evecs 20 --save_name "solenoid_m_3" --simple_out \
        2>&1 | sed -u 's/^/[Node 0] /'  # <--- 여기가 핵심
    done
) &

(
    for s in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95
    do
        echo "Running sigma=$s" | sed -u 's/^/[Node 1] /'
        
        numactl --cpunodebind=1 --membind=1 \
        uv run main.py --n_sites 27 --n_fermions 9 --sigma $s \
           --num_evecs 20 --save_name "solenoid_m_3" --simple_out \
        2>&1 | sed -u 's/^/[Node 1] /'  # <--- 여기가 핵심
    done
) &

wait
echo "All done!"