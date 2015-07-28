from dolfin import *
import hybridmc as hmc

import sys

import sphere3D_1 as sphere
import box3D_1 as box

################################################################
######################    create client    #####################
################################################################

if len(sys.argv) >= 2:
    port = 8888
    timeout = 90  # ****    IMPORTANT    ****
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    if len(sys.argv) >= 4:
        timeout = int(sys.argv[3])

    wsdl_url = "http://%s:%d/?wsdl" %(sys.argv[1],port)
    print wsdl_url
    client = hmc.RemoteClient(wsdl_url)
    client.set_options(timeout=timeout)
else:
    client = hmc.LocalClient()

################################################################

sp = sphere.Problem(client=client,priority=1)
bp = box.Problem(client=client)

config = hmc.IterativeSolverConfig.Config3D()

subdomains=[ sp, bp ]

solutions = hmc.IterativeSolver(subdomains=subdomains,config=config)

for s, d in zip(solutions,subdomains):
    plot(s, title=d.__module__)

interactive()
