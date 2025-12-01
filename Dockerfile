FROM python:3.8-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswresample-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
    
USER user

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==1.10.0+cpu \
        torchvision==0.11.1+cpu \
        torchaudio==0.10.0+cpu \
        -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install pyannote.audio

COPY --chown=user ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
