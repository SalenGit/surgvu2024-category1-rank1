FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=algorithm:algorithm ./ /opt/algorithm/
RUN python -m pip install --user -r requirements_.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENV MKL_THREADING_LAYER='GNU'
ENTRYPOINT python -m process $0 $@ "$(which python)"
