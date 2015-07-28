from dolfin import *
import hybridmc.IterativeSolverConfig as conf

##############################################
###                 Toolbox                ###
##############################################

def Laplacian(expr,x,y,z):
    # find the Laplace operator
    # http://en.wikipedia.org/wiki/Laplace_operator

    dx = diff(expr,x)
    dx2 = diff(dx,x)

    dy = diff(expr,y)
    dy2 = diff(dy,y)

    dz = diff(expr,z)
    dz2 = diff(dz,z)

    dx2dy2dz2 = dx2 + dy2 + dz2

    return dx2dy2dz2

##############################################
###        API for the solver (WIP)        ###
##############################################

# Box's edge points
_x = [ -2, 2 ]
_y = [ -1, 1 ]
_z = [ -.5, .5 ]

# Sphere's center and radius
_c = [ -3, 1, 1 ]
_r = 4

resC = 32

def ExtBC(x,on_boundary):
    return on_boundary and not (    between(x[0],(_x[0],_x[1]))
                                and between(x[1],(_y[0],_y[1]))
                                and between(x[2],(_z[0],_z[1])))

def ExtIface(x,on_boundary):
    return on_boundary and (    between(x[0],(_x[0],_x[1]))
                            and between(x[1],(_y[0],_y[1]))
                            and between(x[2],(_z[0],_z[1])))

class Problem(conf.ConfigCommonProblem):
    def init(self,*args,**kwargs):
        mesh_filename = kwargs.get('mesh_filename')
        mesh = None
        if not mesh_filename:
            # the user creates a custom mesh inside this method
            domain = Sphere(Point(_c[0],_c[1],_c[2]),_r)
            mesh = Mesh(domain,resC)
        else:
            mesh = Mesh(mesh_filename)

        _ex = [ -4, 4 ]
        _ey = [ -2, 2 ]
        _ez = [ -1, 1 ]

        x = variable(Expression("x[0]"))
        y = variable(Expression("x[1]"))
        z = variable(Expression("x[2]"))

        self.reference_function = (x-_ex[0])*(x-_ex[1])*(y-_ey[0])*(y-_ey[1])*(z-_ez[0])*(z-_ez[1])

        self.V = FunctionSpace(mesh,'Lagrange',1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        f = -Laplacian(self.reference_function,x,y,z)

        self.a = inner(grad(u), grad(v))*dx
        self.L = f*v*dx

    def neighbors(self):
        interface = {}
        interface['box3D_1'] = ExtIface
        return interface

    def boundaries(self):
        fixed_bc_expr = self.reference_function
        bc = DirichletBC(self.V, fixed_bc_expr, ExtBC)
        return [ bc ]
