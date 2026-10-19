# 应用镜像：拼装 daemon 二进制 + 前端。（与 moss-tts 同构，只多了常驻 daemon 的启动）
FROM moss-tts-runtime-base:cu128

# 编译产物（llama-moss-tts + libggml*.so / libllama.so）
COPY build/bin /opt/llama/bin
ENV LD_LIBRARY_PATH=/opt/llama/bin
ENV PATH=/opt/llama/bin:/opt/venv/bin:$PATH

WORKDIR /app
COPY server.py /app/server.py
COPY static /app/static
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 静态 ffmpeg（eugeneware/ffmpeg-static b6.1.1，77 MB，静态链接，带 libopus）
# 唯一用途：`stream=true` + response_format=opus 的**真流式**编码（Ogg 页 20 ms）。
# 为什么不用 opusenc：它要等 stdin 关闭才开始写（实测 0 字节直到 EOF），没法边生成边发。
# 校验和固定，换版本必须同时改这里；许可证是 GPL（自用部署，不对外分发二进制）。
COPY docker/bin/ffmpeg /usr/local/bin/ffmpeg
RUN echo "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99  /usr/local/bin/ffmpeg" \
    | sha256sum -c - \
 && chmod +x /usr/local/bin/ffmpeg \
 && ffmpeg -version | head -1

# 常驻 daemon 的 unix socket 目录（compose 里挂出来，宿主也能连，方便调试）
RUN mkdir -p /run/moss

EXPOSE 8080
# entrypoint：先起常驻 C++ daemon，再起 FastAPI 前端（两个进程同一个容器）
ENTRYPOINT ["/app/entrypoint.sh"]
