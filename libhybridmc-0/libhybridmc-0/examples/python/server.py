from hybridmc import server

import sys

# default values
address="127.0.0.1"
port = 8888

if len(sys.argv) >= 2:
    address = argv[1]
if len(sys.argv) >= 3:
    port = int(sys.argv[2])

server.run(address=address,port=port)
