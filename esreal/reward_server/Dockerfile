FROM nvcr.io/nvidia/tritonserver:22.08-py3

WORKDIR /app

COPY requirements.txt .

WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -c "import open_clip;open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')"
RUN pip3 install --no-cache-dir salesforce-lavis
RUN python3 -c "from lavis.common.dist_utils import download_cached_file; download_cached_file('https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth', check_hash=False, progress=True)"
RUN pip3 install --no-cache-dir tomesd
RUN pip3 install --no-cache-dir ultralytics
RUN pip3 install --no-cache-dir opencv-python-headless==4.8.1.78

# fix torchvision version to match torch
RUN pip3 install --no-cache-dir torchvision==0.15.2

# to prevent https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN pip3 uninstall opencv-python opencv-python-headless -y
RUN pip3 install --no-cache-dir opencv-python-headless==4.8.1.78

# Grounding DINO
ENV CUDA_HOME /usr/local/cuda
COPY GroundingDINO /app/GroundingDINO
WORKDIR /app/GroundingDINO
RUN pip3 install -e .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# fix transformer version issue
RUN pip3 install --no-cache-dir transformers==4.34.1

WORKDIR /app
CMD ["tritonserver", "--model-repository=/models", "--model-load-thread-count 8"]