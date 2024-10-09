cd esreal/reward_server

srun sh build_tritonserver.sh

srun --gres=gpu:8 --cpus-per-task=256 sh run_tritonserver.sh

tritonserver --model-repository=/models --model-load-thread-count 8 --log-verbose=1
