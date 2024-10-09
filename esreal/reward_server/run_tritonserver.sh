docker run -it --rm --gpus '"device=0,1,2,3,4,5,6,7"' --cpus=256 --memory=160g --shm-size=80g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v `pwd`/model_repository:/models \
    -v `pwd`/../../models:/app/models \
    triton:23.09-py3-cu1201 \
    bash
    # tritonserver --model-repository=/models --model-load-thread-count 8