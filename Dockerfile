FROM python:3.10-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH 

WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

USER user
RUN pip install --no-cache-dir "pip<24.1"


RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git \
     --force-reinstall
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --force-reinstall
RUN pip install --no-cache-dir \
    pyannote.audio==3.1.1 \
    transformers==4.40.1 
RUN pip install --no-cache-dir "numpy<2" 
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
