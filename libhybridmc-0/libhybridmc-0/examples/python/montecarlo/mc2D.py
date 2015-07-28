from dolfin import *
import hybridmc as hmc

def Laplacian(expr,x,y):
    # find the Laplace operator
    # http://en.wikipedia.org/wiki/Laplace_operator

    dx = diff(expr,x)
    dx2 = diff(dx,x)

    dy = diff(expr,y)
    dy2 = diff(dy,y)

    dx2dy2d = dx2 + dy2

    return dx2dy2d

# def _near(x,t):
#     e = 10e-3
#     return between(x,(t-e,t+e))

def onbc(x,on_boundary):
    return on_boundary

# def iface_Gamma(x,on_boundary):
#     return    (_near(x[0],_x2[0]) and between(x[1],(_y2[0],_y2[1]))
#             or _near(x[0],_x2[1]) and between(x[1],(_y2[0],_y2[1]))
#             or _near(x[1],_y2[0]) and between(x[0],(_x2[0],_x2[1]))
#             or _near(x[1],_y2[1]) and between(x[0],(_x2[0],_x2[1])))

def test(res,logfile,iterations,OpenCL,mpi_workers):
    x = variable(Expression("x[0]"))
    y = variable(Expression("x[1]"))

    expr = (x)*(x-1)*(y)*(y-1)
    f = -Laplacian(expr,x,y)

    mesh = Mesh(Domain,res)
    V = FunctionSpace(mesh,'Lagrange',1)

    u = TrialFunction(V)
    v = TestFunction(V)

    # f_ref = '(x[0]-1)*(x[0])*(x[1]-1)*(x[1])'
    # q_ref = '-1 * (2 * (-1 + (x[0])) * (x[0]) + 2 * (x[1]) * (-1 + (x[1])))'
    f_expr = hmc.tools.cppcode(expr,x,y)
    q_expr = hmc.tools.cppcode(f,x,y)
    print q_expr
    print f_expr

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    bc = DirichletBC(V,expr,onbc)

    sol = Function(V)
    solve(a==L,sol,[ bc ])

    for iteration in range(iterations):
        print "iteration ", iteration

        mcbc, est = client.montecarlo(V, onbc,
                                      OpenCL=OpenCL,
                                      Omega=KnownDomain,
                                      f=f_expr, q=q_expr,
                                      walks=5000, btol=1e-13, threads=6,
                                      mpi_workers=mpi_workers)
        sol_mc = Function(V)
        solve(a==L,sol_mc,[ mcbc ])
        diff_sol = hmc.tools.sol_errornorm(sol,sol_mc)
        diff_bc = hmc.tools.bc_errornorm(bc,mcbc)

        msg = "%d\t%f\t%f\n" % (iteration,diff_bc,diff_sol)
        print diff_bc, diff_sol
        with open(filename, "a") as logfile: logfile.write(msg)

    if False:
        plot(bc,title='bc')
        plot(mcbc,title='bc monte carlo')

        plot(sol,title='solution',scale=0.0)
        plot(sol_mc,title='solution monte carlo',scale=0.0)

        interactive()

    return diff_bc, diff_sol

###########################################
if __name__ == '__main__':
    import sys

    OpenCL = False
    mpi_workers = 5

    # full domain \Omega with \Gamma internal interface
    _x1 = [ 0., 1. ]
    _y1 = [ 0., 1. ]

    # subdomain D with \Gamma external interface
    _x2 = [ .4, .8 ]
    _y2 = [ .4, .8 ]

    KnownDomain = [ _x1[1], _y1[1] ]
    Domain = Rectangle(_x2[0],_y2[0],_x2[1],_y2[1])

    print KnownDomain
    print Domain

    #print R.first_corner().x()
    #print R.first_corner().y()
    #print R.second_corner().x()
    #print R.second_corner().y()

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

    iterations = 1
    for i in range(6,7):
        res = 2**i
        filename = "montecarlo_res%d.log" %(res)
        header = "# %s\n# iter\terror(bc)\terror(sol)\n" %(filename)
        with open(filename, "w") as logfile: logfile.write(header)
        test(res,logfile,iterations,OpenCL,mpi_workers)
