#pragma once

// What TouchDesigner gets about the sound, measured on the samples as they are output
// (so the visuals follow what is heard, not what arrived).
//
// FeatureTap: the audio thread writes each output block (mixed to mono) into a ring;
//             the analysis thread reads the newest samples. One writer, no locks.
// Features:   loudness (RMS dB, 0-1 level, peak), log-spaced frequency bands (0-1),
//             spectral flux, and onsets (hits) picked from the flux.

#include <atomic>
#include <cstdint>
#include <vector>

#include <juce_dsp/juce_dsp.h>

namespace sc
{
class FeatureTap
{
public:
    void prepare (double sampleRate);
    void push (const float* left, const float* right, int frames);  // audio thread
    // copies the newest `n` samples (oldest first) into `out`; false before n were written
    bool latest (float* out, int n) const;
    double sampleRate() const { return rate.load(); }

private:
    static constexpr int capacity = 1 << 15;
    std::vector<float> ring = std::vector<float> (capacity);
    std::atomic<int64_t> written { 0 };
    std::atomic<double> rate { 48000.0 };
};

struct FeatureFrame
{
    float rmsDb = -100, level = 0, peak = 0;
    std::vector<float> bands;  // 0-1 each, low to high
    float flux = 0;            // 0-1-ish, how much the spectrum just grew
    float onset = 0;           // the flux on a frame where a hit was detected, else 0
};

class Features
{
public:
    static constexpr int fftOrder = 11;  // 2048 samples, ~43 ms at 48 kHz
    static constexpr int fftSize = 1 << fftOrder;

    void prepare (double sampleRate, int numBands);
    // `samples`: the newest fftSize output samples; `dt`: seconds since the last call
    FeatureFrame analyse (const float* samples, double dt);

private:
    juce::dsp::FFT fft { fftOrder };
    std::vector<float> window, work, magnitude, previous;
    std::vector<int> bandEdges;  // FFT bin where each band starts (numBands + 1 entries)
    std::vector<float> fluxHistory;
    size_t fluxAt = 0;
    double sinceOnset = 1.0;
    float lastFlux = 0;
};
} // namespace sc
