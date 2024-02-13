FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /tmp
COPY requirements.txt .

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && \
git clone https://github.com/NVIDIA/NeMo.git && pip install --upgrade pip && pip install git+https://github.com/NVIDIA/NeMo.git 

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY utils/* /app/utils/
COPY *.py /app/
RUN chmod +x align.py
COPY stt_en_fastconformer_hybrid_large_pc.nemo /root/.cache/torch/NeMo/NeMo_1.23.0rc0/stt_en_fastconformer_hybrid_large_pc/465b32000fc320f5905fda11a1866ef6/stt_en_fastconformer_hybrid_large_pc.nemo
EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
