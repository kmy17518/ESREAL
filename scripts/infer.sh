srun --gres=gpu:8 --cpus-per-task=128 \
    accelerate launch \
    --multi_gpu --num_processes 8 --main_process_port 10000 \
    esreal/infer.py
