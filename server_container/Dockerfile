# syntax=docker/dockerfile:1.7
#
# The music server for lh2300. Checkpoints stay on scratch (MAGENTA_HOME);
# this image is the JAX process. `make server-singularity` binds that path.

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/superconductor_server
COPY requirements_lighthouse.txt ./
RUN pip install --no-cache-dir -r requirements_lighthouse.txt
COPY mrt2_server.py tempo.py memlog.py ./
COPY pages ./pages

ENV MAGENTA_HOME=/scratch/aimusic_project_root/aimusic_project/shared_data \
    BACKEND=jax \
    MODEL=mrt2_base \
    PORT=9100

EXPOSE 9100
ENTRYPOINT ["sh", "-c", "exec python mrt2_server.py --backend \"$BACKEND\" --model \"$MODEL\" --port \"$PORT\""]
