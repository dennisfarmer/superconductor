// Persistent Bungee Basic streaming bridge. Upstream is licensed MPL-2.0.
#include <bungee/Stream.h>
#include <array>
#include <cmath>
#include <exception>
#include <string>
#include <stdexcept>
#include <vector>

namespace {
thread_local std::string error;
struct Engine {
    Bungee::Stretcher<Bungee::Basic> stretcher;
    Bungee::Stream<Bungee::Basic> stream;
    std::array<std::vector<float>, 2> input, output;
    Engine() : stretcher({48000, 48000}, 2), stream(stretcher, 256, 2) {
        for (auto &v : input) v.resize(256);
        for (auto &v : output) v.resize(5121);
    }
};
}
extern "C" {
const char *conductor_error() { return error.c_str(); }
void *conductor_create() {
    try { return new Engine; }
    catch (const std::exception &e) { error = e.what(); return nullptr; }
}
void conductor_destroy(void *engine) { delete static_cast<Engine *>(engine); }
int conductor_process(void *engine, const float *input, float *output, double speed) {
    try {
        if (!engine || !input || !output || !std::isfinite(speed) || speed < 0.05 || speed > 20)
            throw std::runtime_error("Invalid Bungee stream arguments");
        auto &e = *static_cast<Engine *>(engine);
        for (int i = 0; i < 256; ++i)
            for (int c = 0; c < 2; ++c) e.input[c][i] = input[2*i+c];
        const float *in[] = {e.input[0].data(), e.input[1].data()};
        float *out[] = {e.output[0].data(), e.output[1].data()};
        const int count = e.stream.process(in, out, 256, 256.0 / speed, 1.0);
        for (int i = 0; i < count; ++i)
            for (int c = 0; c < 2; ++c) output[2*i+c] = e.output[c][i];
        return count;
    } catch (const std::exception &e) { error = e.what(); return -1; }
}
}
