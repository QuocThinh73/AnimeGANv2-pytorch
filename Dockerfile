FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip && pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118  && pip3 install --no-cache-dir -r requirements.txt

COPY . /workspace

RUN mkdir -p /workspace/output

ENTRYPOINT ["python3", "train.py"]