import socket
import json

def tcp_server(host, port, signals, handle_func):
    """Wait on a message from a socket OR a shutdown signal."""
    # Create an INET, STREAMing socket, this is TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        # Bind the socket to the server
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen()

        # Socket accept() will block for a maximum of 1 second.  If you
        # omit this, it blocks indefinitely, waiting for a connection.
        sock.settimeout(1)

        while not signals["shutdown"]:

            # Wait for a connection for 1s.  The socket library avoids
            # consuming CPU while waiting for a connection.
            try:
                clientsocket, _ = sock.accept()
            except socket.timeout:
                continue

            # Socket recv() will block for a maximum of 1 second.  If you omit
            # this, it blocks indefinitely, waiting for packets.
            clientsocket.settimeout(1)

            # Receive data, one chunk at a time.  If recv() times out before
            # we can read a chunk, then go back to the top of the loop and try
            # again.  When the client closes the connection, recv() returns
            # empty data, which breaks out of the loop.  We make a simplifying
            # assumption that the client will always cleanly close the
            # connection.
            with clientsocket:
                message_chunks = []
                while True:
                    try:
                        data = clientsocket.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    message_chunks.append(data)

            # Decode list-of-byte-strings to UTF8 and parse JSON data
            message_bytes = b''.join(message_chunks)
            message_str = message_bytes.decode("utf-8")

            try:
                message_dict = json.loads(message_str)
            except json.JSONDecodeError:
                continue
            handle_func(message_dict)


def tcp_client(host, port, message):
    """Test TCP Socket Client."""
    # create an INET, STREAMing socket, this is TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        # connect to the server
        sock.connect((host, port))

        # send a message
        message = json.dumps(message)
        sock.sendall(message.encode('utf-8'))


def udp_client(host, port, message):
    """Test TCP Socket Client."""
    # create an INET, STREAMing socket, this is TCP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

        # connect to the server
        sock.connect((host, port))

        # send a message
        message = json.dumps(message)
        sock.sendall(message.encode('utf-8'))


def udp_server(host, port, signals, handle_func):
    """Test UDP Socket Server."""
    # Create an INET, DGRAM socket, this is UDP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

        # Bind the UDP socket to the server
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.settimeout(1)

        # No sock.listen() since UDP doesn't establish connections like TCP

        # Receive incoming UDP messages
        while not signals["shutdown"]:
            try:
                message_bytes = sock.recv(4096)
            except socket.timeout:
                continue
            message_str = message_bytes.decode("utf-8")
            message_dict = json.loads(message_str)
            handle_func(message_dict)

#"""Example of waiting on a socket or a shutdown signal."""
# def main():
    # """Main thread, which spawns a second server() thread."""
    # print("main() starting")
    # signals = {"shutdown": False}

    # thread = threading.Thread(
    #   target=tcp_server,
    #   args=("localhost", 6000, signals, lambda x:1)
    # )
    # thread.start()
    # time.sleep(10) # Give up execution to the 'server' thread (see Pitfall 1)
    # signals["shutdown"] = True  # Tell server thread to shut down
    # thread.join()  # Wait for server thread to shut down
    # print("main() shutting down")

# if __name__ == "__main__":
    # main()
