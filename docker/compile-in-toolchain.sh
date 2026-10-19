#!/usr/bin/env bash
# 在工具链底包内执行：应用 patches/ -> 增量编译 -> 输出 ccache 统计
# 由 scripts/build-llama.sh 生成并挂进容器，不要单独运行。
set -e

for p in /patches/*.patch; do
  echo ">>> applying $(basename "$p")"
  patch -p1 --forward --batch < "$p"
done

cmake -S /src -B /build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-120} \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF

cmake --build /build -j"${JOBS}" --target llama-moss-tts llama-quantize

echo "--- ccache 统计 ---"
ccache -s 2>/dev/null | sed -n '1,14p' || true
