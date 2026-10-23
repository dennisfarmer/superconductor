#pragma once

// The connection to the client's WS /audio (AUDIO_PROTOCOL.md).
//
//   in:  binary [uint32 LE seq][float32 LE interleaved stereo, 48 kHz]   -> AudioQueue
//        {"type": "role", "role": "clock" | "monitor", "clock": name}
//        {"type": "pause"}
//   out: {"type": "ack"}       (clock only) one per block the audio thread finished
//        {"type": "underrun"}  (clock only) once per stretch of silence
//
// IXWebSocket runs the socket on its own thread and reconnects by itself; a small
// sender thread turns the audio thread's counters into acks, so the audio thread
// never touches the network.

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include <ixwebsocket/IXWebSocket.h>

#include "AudioQueue.h"

namespace sc
{
class StreamClient
{
public:
    explicit StreamClient (AudioQueue& queue);
    ~StreamClient();

    // (re)connects; `wantsClock` asks for role=clock, otherwise role=monitor
    void connect (const std::string& url, bool wantsClock, const std::string& name);
    void disconnect();

    // audio thread
    void noteUnderrun() { underruns.fetch_add (1); }

    // any thread
    bool isConnected() const { return connected.load(); }
    bool isClock() const { return clock.load(); }
    bool isPaused() const { return paused.load(); }  // paused, and nothing new since
    int64_t blocksReceived() const { return received.load(); }
    int64_t underrunCount() const { return underruns.load(); }
    std::string clockName() const;
    std::string lastError() const;

    static constexpr int sampleRate = 48000;

private:
    void onMessage (const ix::WebSocketMessagePtr& msg);
    void onText (const std::string& text);
    void senderLoop();

    AudioQueue& queue;
    ix::WebSocket socket;
    std::thread sender;
    std::atomic<bool> running { true };

    std::atomic<bool> connected { false }, clock { false }, paused { false };
    std::atomic<bool> resyncAcks { false };
    std::atomic<int64_t> received { 0 }, underruns { 0 };

    mutable std::mutex textLock;
    std::string clockNameText, errorText;
    std::vector<float> scratch;  // aligned copy of a block (socket thread only)
};
} // namespace sc
