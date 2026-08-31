# braid, in a container.
#
# Two stages and two toolchains, because the two halves of this server cannot be
# built by the same compiler: the worker is CUDA C++ and the server is Go, and
# the whole reason they are separate processes is that nvcc and cgo do not link
# together. The image inherits that split rather than hiding it.
#
# The runtime image needs a GPU. It is built on nvidia/cuda's *runtime* flavour
# rather than *devel* -- the toolkit is a build-time dependency and carrying it
# into production would add gigabytes nothing runs -- so it wants the NVIDIA
# Container Toolkit on the host:
#
#   docker build -t braid .
#   docker run --gpus all -p 8080:8080 braid
#
# Without --gpus the worker starts, reports "cuda no", and serves from the CPU
# at a speed that is not worth measuring. That is a real fallback rather than a
# crash, and the log line says which one you got.
#
# The model is NOT baked in. braid_train writes a 43 MB checkpoint that is
# gitignored and reproducible from a seed, so an image carrying one would be
# shipping a build artifact with a copy of two repositories' source inside it.
# Mount it:
#
#   docker run --gpus all -v "$PWD/models:/models:ro" -p 8080:8080 braid
#
# ---------------------------------------------------------------------------

# --- stage 1: the worker ----------------------------------------------------
#
# devel, not runtime: nvcc lives here and nowhere else.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS worker

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# The engine is a pinned submodule and the build reads it, so the whole tree
# comes across. Copying only engine/ would produce a configure error about a
# missing subdirectory, which is a worse way to learn the same thing.
COPY engine/ engine/
COPY third_party/ third_party/

# CMAKE_CUDA_ARCHITECTURES is deliberately not pinned to one card. `native`
# would compile for whatever GPU the *builder* has, which is not the one the
# image runs on; this list covers Turing through Ada and costs build time rather
# than correctness.
RUN cmake -B build -S engine -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    && cmake --build build --target braid_worker braid_train

# --- stage 2: the server ----------------------------------------------------
FROM golang:1.27-bookworm AS server

WORKDIR /src
# go.mod has no dependencies, so there is no module cache worth a separate
# layer. If that ever stops being true, split the download from the build.
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/

# CGO_ENABLED=0 for the same reason the worker is a separate process: this
# binary must not want a C toolchain at runtime. Static, so it runs on the CUDA
# runtime image without a matching libc.
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/braid ./cmd/braid \
    && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/braidload ./cmd/braidload

# --- stage 3: what actually ships -------------------------------------------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# libgomp for the engine's CPU fallback paths, which the worker uses for
# anything below its dispatch thresholds and for every operation if there is no
# GPU. Without it the worker exits at load with a link error that says nothing
# about GPUs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home braid

COPY --from=worker /src/build/braid_worker /usr/local/bin/braid_worker
COPY --from=worker /src/build/braid_train  /usr/local/bin/braid_train
COPY --from=server /out/braid              /usr/local/bin/braid
COPY --from=server /out/braidload          /usr/local/bin/braidload

USER 10001:10001

EXPOSE 8080

# 0.0.0.0 with no -auth-token makes the server refuse to start, which is the
# behaviour this project chose over a README sentence asking people to be
# careful. In a container that refusal is the useful default: an image that
# listens on every interface unauthenticated because it was convenient is how
# this goes wrong. Pass -auth-token, or override the command with -addr
# 127.0.0.1:8080 and reach it another way.
ENTRYPOINT ["braid"]
CMD ["-addr", "0.0.0.0:8080", "-model", "/models/charlm", "-worker", "/usr/local/bin/braid_worker"]

# healthz is the liveness answer and it is deliberately not a generation: a
# check that ran the model would report a busy server as a dead one.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD ["/usr/local/bin/braidload", "-addr", "http://127.0.0.1:8080", "-requests", "0"]
