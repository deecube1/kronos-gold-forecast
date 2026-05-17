FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone Kronos repo
RUN git clone https://github.com/shiyu-coder/Kronos.git /tmp/Kronos

# Pre-download model weights into image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='/app/models/tokenizer'); snapshot_download('NeoQuasar/Kronos-base', local_dir='/app/models/kronos-base')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
