# Diffusion Gufo: DiffusionGemma on Strix Halo

This is a fork of [Gufo](https://github.com/gufo-org/gufo), the Strix Halo
inference engine, that adds Google's
[DiffusionGemma 26B-A4B](https://huggingface.co/google/diffusiongemma-26B-A4B-it):
a Gemma 4 mixture-of-experts model that writes text by block diffusion. It
runs as a native HIP engine on AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S,
`gfx1151`) and is served by `gufo serve llm` with an OpenAI-compatible API. For
everything else Gufo does, see the
[upstream README](https://github.com/gufo-org/gufo#readme).

I may submit this fork to gufo, if so this fork will not be maintained.

## What this fork adds

- **Engine** (`src/models/diffusion_gemma/`): a causal encoder for the prompt
  cache and a bidirectional denoiser over 256-token canvases, with
  self-conditioning and the checkpoint's entropy-bound sampler. Every kernel
  is Gufo's own: dense WMMA products, flash attention for head dimensions 256
  and 512, grouped expert GEMMs and a fused sampler. There is no tuning step,
  and repeated runs are bit-identical.
- **Checkpoints, loaded directly:** the original BF16 safetensors, RedHatAI
  FP8, NVIDIA NVFP4 and Unsloth Q8_0 GGUF. Quantized experts stay quantized on
  the GPU and are widened to BF16 inside the expert kernels, because Strix
  Halo has no FP8 or FP4 matrix units.
- **Server** (`src/cli/serve/diffusion_gemma_backend.*`): Chat Completions,
  Responses and completions, streamed one canvas at a time, with function
  tools, prompt-prefix reuse, optional thinking
  (`chat_template_kwargs.enable_thinking`), and `ignore_eos` / `cache_prompt`
  on raw completions.
- **Docs:** [usage](docs/models/diffusiongemma/README.md),
  [benchmarks](docs/models/diffusiongemma/BENCHMARKS.md),
  [quality](docs/models/diffusiongemma/QUALITY.md) and
  [kernel experiments](docs/models/diffusiongemma/EXPERIMENTS.md).

## Results

Radeon 8060S, ROCm 7.2.3, through the HTTP server, one request at a time. The
reference is llama.cpp's DiffusionGemma server with the Q8_0 GGUF (llama.cpp
has no FP8 type).

Prefill, tok/s, book prompts:

| Prompt tokens | BF16 | FP8 | NVFP4 | llama.cpp Q8_0 |
| ---: | ---: | ---: | ---: | ---: |
| 2,261 | 2,901 | 3,116 | 3,443 | 978 |
| 33,105 | 2,387 | 2,326 | 2,537 | 445 |
| 121,294 | 1,344 | 1,329 | 1,379 | 201 |

Generation, 508-token prompt and 512 tokens out:

| | BF16 | FP8 | NVFP4 | llama.cpp Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| tok/s, whole request | 52.7 | 63.7 | 81.8 | 20.7 |
| ms per denoising step | 182 | 136 | 116 | 386 |

On the SPEED-Bench prompts that
[diffu-benching](https://github.com/iv2nl0b9v/diffu-benching) ran on an NVIDIA
DGX Spark (GB10, vLLM, NVFP4), Gufo NVFP4 reaches 77–103% of the Spark's
generation speed per user, with the gap closing as prompts get longer.

Accuracy against Hugging Face Transformers on the original BF16 checkpoint
(mean KL per canvas position, first denoising step) and MMLU (all 14,042
questions, 5-shot, thinking off):

| | BF16 | FP8 | NVFP4 |
| --- | ---: | ---: | ---: |
| KL to the reference (noise floor 0.012–0.018) | 0.022 | 0.029 | 0.070 |
| MMLU | – | 81.1% | 80.7% |

## Quickstart

You need a Strix Halo running Linux with ROCm 7.2.3. Build from source:
upstream Gufo's prebuilt images do not include DiffusionGemma.

**1. Build.** The commands below use Debian/Ubuntu package names; see
[Build from source](#build-from-source) for other systems or Nix.

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev ffmpeg \
  hipblas-dev hipblaslt-dev rocblas-dev hipcub-dev rocprim-dev rocwmma-dev
git clone https://github.com/robotsbuildrobots/diffusion-gufo
cd diffusion-gufo
cmake --preset release
cmake --build --preset release --parallel 8
./build/release/gufo diagnose
```

**2. Download weights.** NVFP4 is the fastest and smallest (about 19 GB). The
FP8 and BF16 downloads are in the
[model guide](docs/models/diffusiongemma/README.md#weights). `hf` comes with
`pip install -U huggingface_hub`.

```sh
hf download nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --revision ec4ff3df205028f4e81c954c2227f9312b3ec2ea \
  --local-dir models/diffusiongemma-nvfp4
```

**3. Serve.** Point `--model` at the checkpoint directory. Startup takes about
10 s from a warm page cache. Stop other GPU servers first.

```sh
./build/release/gufo serve --port 8080 llm --model models/diffusiongemma-nvfp4 \
  --served-model-name diffusiongemma --context 131072
```

**4. Send a request.** Any OpenAI-compatible client works. Thinking is off
unless the request turns it on:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","messages":[{"role":"user","content":"Why is the sky blue?"}]}'

curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","chat_template_kwargs":{"enable_thinking":true},
       "messages":[{"role":"user","content":"What is 17*23?"}]}'
```

Text arrives a 256-token canvas at a time, and requests run one at a time.
`temperature` and `top_p` do not apply, because the diffusion sampler
replaces them.

## Build from source

Linux x86-64 on AMD Strix Halo (`gfx1151`) is the supported target. CMake owns
one production configuration for both Nix and ordinary Linux builds. Tests,
profilers, tuning executables and Python reference runners are not installed
with the production package. No `build.sh` wrapper is needed.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

[flake.lock](flake.lock) pins the dependencies. `nix develop` adds profiling,
model-download and independent evaluation tools; these are not runtime
requirements. Optional benchmark baselines are selected separately with
`nix shell .#ds4-reference`, `.#llama-cpp-reference` or
`.#llama-cpp-mtp-reference`; see [benchmarking](docs/BENCHMARKS.md).
See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the
following development libraries. The currently qualified toolchain is GCC
15.3 and ROCm 7.2.3. Attention and audio convolution kernels are compiled
directly from HIP. Python, Triton/AOTriton, Composable Kernel and MIOpen are
not production build or runtime requirements.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA headers | Compiled GPU kernels |
| ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp | Tokenization, HTTPS, hashing and images |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. ROCm normally installs
under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev ffmpeg

# ROCm libraries from the table, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Configuring fails at `find_package(hipblas)` when those ROCm packages are
missing. Other distributions name them `-devel` instead of `-dev`.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm"`.
If compiler discovery picks a system Clang, also pass
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`.
Use `cmake --install build/release` to install Gufo,
its runtime data and license notices. The GPU driver must allow your user to
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately.

The same source, compiler flags and install rules serve both builds. Nix pins
the complete toolchain for reproducible comparisons; changing the compiler or
math libraries requires the affected model's quality checks.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`, installed under `share/licenses/gufo`. Model weights are not
bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
