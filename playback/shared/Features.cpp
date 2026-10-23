#include "Features.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sc
{
namespace
{
constexpr float minDb = -80.0f;
constexpr double lowestBandHz = 40.0, highestBandHz = 16000.0;
constexpr int rmsSamples = 1024;          // ~21 ms at 48 kHz
constexpr size_t fluxHistoryLength = 30;  // ~0.5 s at 60 analyses a second
constexpr float onsetFactor = 1.5f, onsetFloor = 0.02f;
constexpr double minOnsetGap = 0.08;  // s

float toDb (float x) { return x > 0 ? std::max (minDb, 20.0f * std::log10 (x)) : minDb; }
float dbTo01 (float db) { return std::clamp ((db - minDb) / -minDb, 0.0f, 1.0f); }
} // namespace

void FeatureTap::prepare (double sampleRate)
{
    rate = sampleRate;
}

void FeatureTap::push (const float* left, const float* right, int frames)
{
    auto w = written.load (std::memory_order_relaxed);
    for (int i = 0; i < frames; ++i, ++w)
        ring[(size_t) (w & (capacity - 1))] = 0.5f * (left[i] + right[i]);
    written.store (w, std::memory_order_release);
}

bool FeatureTap::latest (float* out, int n) const
{
    const auto w = written.load (std::memory_order_acquire);
    if (w < n)
        return false;
    // the writer may lap the oldest of these while we copy; the ring is 16x the window,
    // so that only happens if the analysis thread stalls for ~0.6 s
    for (int i = 0; i < n; ++i)
        out[i] = ring[(size_t) ((w - n + i) & (capacity - 1))];
    return true;
}

void Features::prepare (double sampleRate, int numBands)
{
    window.resize (fftSize);
    for (int i = 0; i < fftSize; ++i)
        window[(size_t) i] = 0.5f - 0.5f * std::cos (2.0f * juce::MathConstants<float>::pi * (float) i / (fftSize - 1));
    work.assign (2 * fftSize, 0.0f);
    magnitude.assign (fftSize / 2 + 1, 0.0f);
    previous.assign (fftSize / 2 + 1, 0.0f);

    // log-spaced band edges, each band at least one bin wide
    const auto nyquist = sampleRate / 2;
    const auto top = std::min (highestBandHz, nyquist);
    bandEdges.clear();
    for (int b = 0; b <= numBands; ++b)
    {
        const auto hz = lowestBandHz * std::pow (top / lowestBandHz, (double) b / numBands);
        auto bin = (int) std::round (hz / sampleRate * fftSize);
        if (! bandEdges.empty())
            bin = std::max (bin, bandEdges.back() + 1);
        bandEdges.push_back (std::min (bin, fftSize / 2));
    }

    fluxHistory.assign (fluxHistoryLength, 0.0f);
    fluxAt = 0;
    sinceOnset = 1.0;
    lastFlux = 0;
}

FeatureFrame Features::analyse (const float* samples, double dt)
{
    FeatureFrame f;

    // loudness, over the newest rmsSamples
    double sum = 0;
    float peak = 0;
    for (int i = fftSize - rmsSamples; i < fftSize; ++i)
    {
        sum += (double) samples[i] * samples[i];
        peak = std::max (peak, std::abs (samples[i]));
    }
    f.rmsDb = toDb ((float) std::sqrt (sum / rmsSamples));
    f.level = dbTo01 (f.rmsDb);
    f.peak = peak;

    // spectrum
    std::fill (work.begin(), work.end(), 0.0f);
    for (int i = 0; i < fftSize; ++i)
        work[(size_t) i] = samples[i] * window[(size_t) i];
    fft.performFrequencyOnlyForwardTransform (work.data(), true);
    const auto scale = 2.0f / (float) fftSize;  // a full-scale sine reads ~0.5 (Hann)
    for (size_t k = 0; k < magnitude.size(); ++k)
        magnitude[k] = work[k] * scale;

    const auto numBands = (int) bandEdges.size() - 1;
    f.bands.resize ((size_t) std::max (0, numBands));
    for (int b = 0; b < numBands; ++b)
    {
        float energy = 0;
        for (int k = bandEdges[(size_t) b]; k < bandEdges[(size_t) b + 1]; ++k)
            energy += magnitude[(size_t) k] * magnitude[(size_t) k];
        f.bands[(size_t) b] = dbTo01 (toDb (std::sqrt (energy)));
    }

    // spectral flux: how much the (log) spectrum grew since the last analysis
    double flux = 0;
    for (size_t k = 1; k < magnitude.size(); ++k)
    {
        const auto now = std::log1p (100.0f * magnitude[k]);
        flux += std::max (0.0f, now - previous[k]);
        previous[k] = now;
    }
    f.flux = (float) (flux / (double) magnitude.size());

    // onset: the flux rising, well above its recent average
    const auto average = std::accumulate (fluxHistory.begin(), fluxHistory.end(), 0.0f) / (float) fluxHistory.size();
    sinceOnset += dt;
    if (f.flux > average * onsetFactor + onsetFloor && f.flux >= lastFlux && sinceOnset >= minOnsetGap)
    {
        f.onset = f.flux;
        sinceOnset = 0;
    }
    fluxHistory[fluxAt] = f.flux;
    fluxAt = (fluxAt + 1) % fluxHistory.size();
    lastFlux = f.flux;
    return f;
}
} // namespace sc
