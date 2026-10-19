# voices/

音色（零样本克隆的参考音频）放在这里，**不入库**——它们是个人声音样本，不应该随源码分发。

## 约定

```
voices/<id>.wav                     # 24 kHz / 单声道 / 16-bit
voices/<id>.<wav 的 sha256 前 8 位>.codes.bin   # 预计算的参考 codes
```

`<id>` 只允许 `[A-Za-z0-9_.-]{1,64}`。codes 的指纹由 wav 内容决定，改名/替换音频后必须重算。

## 添加音色

```bash
./scripts/add-voice.sh <源音频> <id>     # 转成 24k mono，并预计算 codes
```

## 没有 codes 会怎样

服务端会在 CPU 上补算一次（`-ngl 0`，约数秒），结果按内容寻址落盘；但首次请求会明显变慢，
所以建议注册时就跑完 `add-voice.sh`。
