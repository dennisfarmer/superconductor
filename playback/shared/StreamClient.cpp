#include "StreamClient.h"

#include <chrono>
#include <cstring>

#include <ixwebsocket/IXNetSystem.h>
#include <juce_core/juce_core.h>

namespace sc
{
StreamClient::StreamClient (AudioQueue& q) : queue (q)
{
    ix::initNetSystem();
    socket.disablePerMessageDeflate();  // float audio doesn't compress
    socket.setMinWaitBetweenReconnectionRetries (250);
    socket.setMaxWaitBetweenReconnectionRetries (2000);
    socket.setOnMessageCallback ([this] (const ix::WebSocketMessagePtr& msg) { onMessage (msg); });
    sender = std::thread ([this] { senderLoop(); });
}

StreamClient::~StreamClient()
{
    running = false;
    sender.join();
    socket.stop();
}

void StreamClient::connect (const std::string& url, bool wantsClock, const std::string& name)
{
    socket.stop();
    const auto sep = url.find ('?') == std::string::npos ? "?" : "&";
    const auto encodedName = juce::URL::addEscapeChars (juce::String (name), true).toStdString();
    socket.setUrl (url + sep + "role=" + (wantsClock ? "clock" : "monitor") + "&name=" + encodedName);
    socket.enableAutomaticReconnection();
    socket.start();
}

void StreamClient::disconnect()
{
    socket.stop();
    connected = false;
    clock = false;
}

std::string StreamClient::clockName() const
{
    std::lock_guard<std::mutex> lock (textLock);
    return clockNameText;
}

std::string StreamClient::lastError() const
{
    std::lock_guard<std::mutex> lock (textLock);
    return errorText;
}

void StreamClient::onMessage (const ix::WebSocketMessagePtr& msg)
{
    switch (msg->type)
    {
        case ix::WebSocketMessageType::Open:
        {
            queue.flush();  // whatever is left from an earlier connection
            connected = true;
            std::lock_guard<std::mutex> lock (textLock);
            errorText.clear();
            break;
        }
        case ix::WebSocketMessageType::Close:
            connected = false;
            clock = false;
            break;
        case ix::WebSocketMessageType::Error:
        {
            connected = false;
            clock = false;
            std::lock_guard<std::mutex> lock (textLock);
            errorText = msg->errorInfo.reason;
            break;
        }
        case ix::WebSocketMessageType::Message:
            if (! msg->binary)
            {
                onText (msg->str);
            }
            else if (msg->str.size() > 4)
            {
                // [uint32 LE seq][float32 LE (n, 2)]; the copy also aligns the floats
                const auto frames = (int) ((msg->str.size() - 4) / (2 * sizeof (float)));
                scratch.resize ((size_t) frames * 2);
                std::memcpy (scratch.data(), msg->str.data() + 4, scratch.size() * sizeof (float));
                queue.push (scratch.data(), frames);
                received.fetch_add (1);
                paused = false;
            }
            break;
        default:
            break;
    }
}

void StreamClient::onText (const std::string& text)
{
    const auto json = juce::JSON::parse (juce::String (text));
    const auto type = json["type"].toString();
    if (type == "role")
    {
        const bool nowClock = json["role"].toString() == "clock";
        if (nowClock && ! clock.load())
        {
            // the blocks not acked yet come again: start from an empty queue
            queue.flush();
            resyncAcks = true;
        }
        clock = nowClock;
        std::lock_guard<std::mutex> lock (textLock);
        clockNameText = json["clock"].isString() ? json["clock"].toString().toStdString() : std::string();
    }
    else if (type == "pause")
    {
        paused = true;  // the silence that follows isn't an underrun
    }
}

void StreamClient::senderLoop()
{
    int64_t acked = queue.blocksPlayed();
    int64_t underrunsSent = underruns.load();
    while (running.load())
    {
        std::this_thread::sleep_for (std::chrono::milliseconds (2));
        const auto played = queue.blocksPlayed();
        const auto under = underruns.load();
        if (resyncAcks.exchange (false) || ! clock.load() || ! connected.load())
        {
            // only blocks played as the clock, on this connection, are acked
            acked = played;
            underrunsSent = under;
            continue;
        }
        for (; acked < played; ++acked)
            socket.sendText (R"({"type": "ack"})");
        if (under > underrunsSent)
        {
            socket.sendText (R"({"type": "underrun"})");
            underrunsSent = under;
        }
    }
}
} // namespace sc
