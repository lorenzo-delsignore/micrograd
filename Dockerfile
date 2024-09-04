FROM python:3.12.5-bookworm

RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update \
  && apt-get install -y --no-install-recommends --fix-missing graphviz \
  && rm -rf /var/lib/apt/lists/*

COPY . /smallgrad
WORKDIR smallgrad
RUN --mount=type=cache,target=/root/.cache \
  pip3 install --upgrade pip && \
  pip3 install --upgrade setuptools wheel && \
  pip3 install smallgrad
ENTRYPOINT ["python", "smallgrad/build_graph.py"]
