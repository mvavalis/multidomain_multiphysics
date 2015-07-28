from dolfin import *
import hybridmc.IterativeSolverConfig as conf

##############################################
###                 Toolbox                ###
##############################################

def Laplacian(expr,x,y):
    # find the Laplace operator
    # http://en.wikipedia.org/wiki/Laplace_operator

    dxexpr = diff(expr,x)
    dx2expr = diff(dxexpr,x)

    dyexpr = diff(expr,y)
    dy2expr = diff(dyexpr,y)

    dx2dy2expr = dx2expr + dy2expr

    return dx2dy2expr

##############################################
###        API for the solver (WIP)        ###
##############################################

# Rectangle's edge points
_x = [ -2, 2 ]
_y = [ -1, 1 ]

# Circle's center and radius
_c = [ -3.5, 0 ]
_r = 4

resC = 32

######################
###  debug method  ###
######################
#def IfaceFromOtherDomain(x,on_boundary):
    ##return near(x[0],_x[0]) or (near(x[1],_y[0]) or near(x[1],_y[1]))
    #t = pi/resC
    #return (
               #(between(x[0],(_x[0]-t,_x[0]+t)) and between(x[1],(_y[0],_y[1])))\
            #or (between(x[0],(_x[1]-t,_x[1]+t)) and between(x[1],(_y[0],_y[1])))\
            #or (between(x[1],(_y[0]-t,_y[0]+t)) and between(x[0],(_x[0],_x[1])))\
            #or (between(x[1],(_y[1]-t,_y[1]+t)) and between(x[0],(_x[0],_x[1])))\
            #)
######################

def ExtBC(x,on_boundary):
    return on_boundary and not (between(x[0],(_x[0],_x[1])) and between(x[1],(_y[0],_y[1])))

def ExtIface(x,on_boundary):
    return on_boundary and (between(x[0],(_x[0],_x[1])) and between(x[1],(_y[0],_y[1])))

class Problem(conf.ConfigCommonProblem):
    def init(self,*args,**kwargs):
        mesh_filename = kwargs.get('mesh_filename')
        mesh = None
        if not mesh_filename:
            # the user creates a custom mesh inside this method
            domain = Circle(_c[0],_c[1],_r)
            mesh = Mesh(domain,resC)
        else:
            mesh = Mesh(mesh_filename)

        _ex = [ -4, 4 ]
        _ey = [ -2, 2 ]
        x = variable(Expression("x[0]"))
        y = variable(Expression("x[1]"))
        self.reference_function = (x-_ex[0])*(x-_ex[1])*(y-_ey[0])*(y-_ey[1])

        self.V = FunctionSpace(mesh,'Lagrange',1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        f = -Laplacian(self.reference_function,x,y)

        self.a = inner(grad(u), grad(v))*dx
        self.L = f*v*dx

    def neighbors(self):
        interface = {}
        interface['rectangle2D_1'] = ExtIface
        return interface

    def boundaries(self):
        fixed_bc_expr = self.reference_function
        bc = DirichletBC(self.V, fixed_bc_expr, ExtBC)
        return [ bc ]
