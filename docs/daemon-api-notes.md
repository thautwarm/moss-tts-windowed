# daemon 要调用的 API 精确签名（从 run-moss-tts-delay.cpp 抠出，供写 moss-tts-daemon.cpp 用）

## struct moss_delay_state / moss_audio_runtime / moss_delay_config / moss_sampling_config
struct moss_sampling_config {
    float text_temperature = 1.5f;
    float text_top_p = 1.0f;
    int32_t text_top_k = 50;
    float audio_temperature = 1.7f;
    float audio_top_p = 0.8f;
    int32_t audio_top_k = 25;
    float audio_repetition_penalty = 1.0f;
};

struct moss_delay_config {
    uint32_t n_vq = MOSS_DELAY_DEFAULT_N_VQ;
    llama_token pad_token_id = MOSS_DELAY_DEFAULT_PAD_TOKEN_ID;
    llama_token im_start_token_id = MOSS_DELAY_DEFAULT_IM_START_TOKEN_ID;
    llama_token im_end_token_id = MOSS_DELAY_DEFAULT_IM_END_TOKEN_ID;
    llama_token audio_start_token_id = MOSS_DELAY_DEFAULT_AUDIO_START_TOKEN_ID;
    llama_token audio_end_token_id = MOSS_DELAY_DEFAULT_AUDIO_END_TOKEN_ID;
    llama_token audio_user_slot_token_id = MOSS_DELAY_DEFAULT_AUDIO_USER_SLOT_TOKEN_ID;
    llama_token audio_assistant_gen_slot_token_id = MOSS_DELAY_DEFAULT_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;
    llama_token audio_assistant_delay_slot_token_id = MOSS_DELAY_DEFAULT_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID;
    llama_token audio_pad_code = MOSS_DELAY_DEFAULT_AUDIO_PAD_CODE;
    uint32_t audio_vocab_size = MOSS_DELAY_DEFAULT_AUDIO_VOCAB_SIZE;

    size_t packed_stride() const {
        return 1u + n_vq;
    }
};

struct moss_audio_segment {
    std::vector<llama_token> codes;
    size_t n_frames = 0;
};

struct moss_generation_audio {
    std::vector<llama_token> delayed_codes;
    size_t delayed_frames = 0;

    std::vector<moss_audio_segment> segments;

    std::vector<llama_token> raw_codes;
    size_t raw_frames = 0;
};

struct moss_prompt_input {
    std::vector<llama_token> packed_ids;
    size_t prompt_frames = 0;
    size_t reference_frames = 0;
};

struct moss_delay_state {
    int32_t audio_length = 0;
    int64_t delayed_length = MOSS_DELAY_INT64_MAX;
    bool is_audio = false;
    bool is_stopping = false;
    int32_t time_step = 0;
    std::vector<llama_token> text_history;

    uint32_t n_vq = MOSS_DELAY_DEFAULT_N_VQ;
    std::vector<llama_token> audio_history;

    size_t audio_frames() const {
        return n_vq == 0 ? 0 : audio_history.size() / n_vq;
    }

    bool empty_audio() const {
        return audio_history.empty();
    }

    const llama_token * audio_frame_ptr(size_t frame_idx) const {
        if (n_vq == 0 || frame_idx >= audio_frames()) {
            return nullptr;
        }
        return audio_history.data() + frame_idx * n_vq;
    }

    void reserve_audio_frames(size_t frames) {
        audio_history.reserve(frames * n_vq);
    }

    void append_audio(const std::vector<llama_token> & frame) {
        GGML_ASSERT(frame.size() == n_vq);
        audio_history.insert(audio_history.end(), frame.begin(), frame.end());
    }

    void append_audio(const llama_token * frame) {
        GGML_ASSERT(frame != nullptr);
        audio_history.insert(audio_history.end(), frame, frame + n_vq);
    }
};

using moss_rng = std::mt19937;

struct moss_codes_header {

## moss_init_delay_state
static moss_delay_state moss_init_delay_state(
        const std::vector<llama_token> & packed_input_ids,
        const moss_delay_config & cfg) {
    GGML_ASSERT(cfg.n_vq > 0);
    GGML_ASSERT(packed_input_ids.size() % cfg.packed_stride() == 0);

    moss_delay_state state;
    state.n_vq = cfg.n_vq;

    const size_t seq_len = packed_input_ids.size() / cfg.packed_stride();
    state.text_history.resize(seq_len);
    state.reserve_audio_frames(std::max<size_t>(seq_len + 1024, 256));

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t row = t * cfg.packed_stride();
        state.text_history[t] = packed_input_ids[row];
        state.audio_history.insert(
                state.audio_history.end(),

## moss_delay_step
static std::vector<llama_token> moss_delay_step(
        moss_delay_state & state,
        const std::vector<float> & text_logits,
        const std::vector<float> & audio_logits,
        const moss_sampling_config & sampling_cfg,
        const moss_delay_config & cfg,
        moss_rng & rng) {
    GGML_ASSERT(cfg.n_vq == state.n_vq);

    const size_t n_vq = cfg.n_vq;
    const size_t text_vocab = text_logits.size();
    const size_t audio_vocab = moss_audio_vocab_with_pad(cfg);
    GGML_ASSERT(audio_logits.size() == n_vq * audio_vocab);


## moss_load_audio_runtime / moss_audio_runtime

    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        throw std::runtime_error("failed to create audio context");
    }

    llama_set_warmup(ctx.get(), false);
    llama_set_causal_attn(ctx.get(), false);
    llama_set_embeddings(ctx.get(), true);

    return ctx;
}

static moss_audio_runtime moss_load_audio_runtime(
        const std::string & model_path,
        const char * expected_arch,
        int32_t n_gpu_layers,
        uint32_t n_ctx) {
    moss_audio_runtime runtime;
    runtime.model = moss_load_audio_model(model_path, expected_arch, n_gpu_layers);
    runtime.ctx = moss_init_audio_context(runtime.model.get(), n_ctx);
    return runtime;
}

static int64_t moss_find_last_equal(const std::vector<llama_token> & values, llama_token target) {
    for (int64_t i = (int64_t) values.size() - 1; i >= 0; --i) {
        if (values[(size_t) i] == target) {
            return i;
        }
    }
    return -1;
}

static moss_delay_state moss_init_delay_state(
        const std::vector<llama_token> & packed_input_ids,
        const moss_delay_config & cfg) {

## moss_decode_audio_llama（解码尾部：codes→波形→wav）
static void moss_decode_audio_llama(
        const std::string & audio_decoder_model_path,
        const std::vector<llama_token> & raw_codes,
        size_t raw_frames,
        const moss_delay_config & cfg,
        int32_t n_gpu_layers,
        const std::string & wav_out_path) {
    moss_audio_runtime runtime = moss_load_audio_runtime(
            audio_decoder_model_path,
            "moss-tts-audio-decoder",
            n_gpu_layers,
            std::max<uint32_t>((uint32_t) raw_frames, 1u));

    const int sample_rate = (int) moss_audio_model_sampling_rate(runtime.model.get(), "moss-tts-audio-decoder");
    const uint32_t downsample_rate = moss_audio_model_downsample_rate(runtime.model.get(), "moss-tts-audio-decoder");
    const uint32_t model_quantizers = moss_audio_model_num_quantizers(runtime.model.get(), "moss-tts-audio-decoder");
    if (cfg.n_vq != model_quantizers) {
        throw std::runtime_error(
                "audio decoder quantizer count mismatch: model expects " +
                std::to_string(model_quantizers) + ", got " + std::to_string(cfg.n_vq));
    }
    if (raw_codes.size() != raw_frames * (size_t) cfg.n_vq) {
        throw std::runtime_error("audio decoder raw code payload size mismatch");
    }

    std::vector<float> audio;
    if (raw_frames > 0) {
        moss_owned_batch batch = moss_batch_from_audio_codes(raw_codes, raw_frames, cfg.n_vq);
        const int ret = llama_encode(runtime.ctx.get(), batch.batch);
        if (ret != 0) {
            throw std::runtime_error("audio decoder llama_encode failed: " + std::to_string(ret));
        }

        const int32_t n_embd_out = llama_model_n_embd_out(runtime.model.get());
        if (n_embd_out != 1) {
            throw std::runtime_error("audio decoder output dimension must be 1");
        }

        const size_t n_samples = raw_frames * (size_t) downsample_rate;
        const float * embd = llama_get_embeddings(runtime.ctx.get());
        if (embd == nullptr) {
            throw std::runtime_error("audio decoder returned null embeddings");
        }
        audio.assign(embd, embd + n_samples);
    }

    if (!save_wav16(wav_out_path, audio, sample_rate)) {
        throw std::runtime_error("failed to write WAV file: " + wav_out_path);
    }
}

static std::vector<float> moss_read_wav_f32_mono(const std::string & path, int expected_sample_rate) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open wav file: " + path);
    }

## moss_extract_audio_segments / moss_concat_audio_segments 签名
static std::vector<moss_audio_segment> moss_extract_audio_segments(
        const std::vector<llama_token> & generation_audio,
        size_t delayed_frames,
        const moss_delay_config & cfg) {
    size_t n_frames = 0;
    const std::vector<llama_token> codes = moss_apply_de_delay_pattern(generation_audio, delayed_frames, cfg, &n_frames);
    if (n_frames == 0) {
        return {};
static std::vector<llama_token> moss_concat_audio_segments(
        const std::vector<moss_audio_segment> & segments,
        size_t n_vq,
        size_t * out_frames = nullptr) {
    size_t total_frames = 0;
    size_t total_tokens = 0;
    for (const auto & seg : segments) {
        total_frames += seg.n_frames;
        total_tokens += seg.codes.size();
