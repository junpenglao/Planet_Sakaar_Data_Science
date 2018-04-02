import theano
import theano.tensor as T
from scipy.integrate import quad
import numpy as np

class integrateOut(theano.Op):
    """
    Integrate out a variable from an expression, computing
    the definite integral w.r.t. the variable specified
    !!! Only implemented in this for scalars !!!


    Parameters
    ----------
    f : scalar
        input 'function' to integrate
    t : scalar
        the variable to integrate out
    t0: float
        lower integration limit
    tf: float
        upper integration limit

    Returns
    -------
    scalar
        a new scalar with the 't' integrated out

    Notes
    -----

    usage of this looks like:
    x = T.dscalar('x')
    y = T.dscalar('y')
    t = T.dscalar('t')

    z = (x**2 + y**2)*t

    # integrate z w.r.t. t as a function of (x,y)
    intZ = integrateOut(z,t,0.0,5.0)(x,y)
    gradIntZ = T.grad(intZ,[x,y])

    funcIntZ = theano.function([x,y],intZ)
    funcGradIntZ = theano.function([x,y],gradIntZ)

    """
    def __init__(self,f,t,t0,tf,*args,**kwargs):
        super(integrateOut,self).__init__()
        self.f = f
        self.t = t
        self.t0 = t0
        self.tf = tf

    def make_node(self,*inputs):
        self.fvars=list(inputs)
        # This will fail when taking the gradient... don't be concerned
        try:
            self.gradF = T.grad(self.f,self.fvars)
        except:
            self.gradF = None
        return theano.Apply(self,self.fvars,[T.dscalar().type()])

    def perform(self,node, inputs, output_storage):
        # Everything else is an argument to the quad function
        args = tuple(inputs)
        # create a function to evaluate the integral
        f = theano.function([self.t]+self.fvars,self.f)
        # actually compute the integral
        output_storage[0][0] = quad(f,self.t0,self.tf,args=args)[0]

    def grad(self,inputs,grads):
        return [integrateOut(g,self.t,self.t0,self.tf)(*inputs)*grads[0] \
            for g in self.gradF]

x = T.dscalar('x')
y = T.dscalar('y')
t = T.dscalar('t')

z = (x**2+y**2)*t
intZ = integrateOut(z,t,0,1)(x,y)
gradIntZ = T.grad(intZ,[x,y])
funcIntZ = theano.function([x,y],intZ)
funcGradIntZ = theano.function([x,y],gradIntZ)
print (funcIntZ(2,2))
print (funcGradIntZ(2,2))

gamma = T.dscalar('gamma')
beta = T.dscalar('beta')
theta = T.dscalar('theta')
z = T.cos(theta)**(2*((gamma/(beta - 2)) - 3/2) + 3)
intZ = integrateOut(z, theta, -(np.pi)/2, (np.pi)/2)(gamma, beta)
gradIntZ = T.grad(intZ, [gamma, beta])
funcIntZ = theano.function([gamma, beta], intZ)
funcGradIntZ = theano.function([gamma, beta], gradIntZ)
print (funcIntZ(3.77,2.5))
print (funcGradIntZ(3.77,2.5))


gamma = 3.77
beta = T.dscalar('beta')
theta = T.dscalar('theta')
z = T.cos(theta)**(2*((gamma/(beta - 2)) - 3/2) + 3)
intZ = integrateOut(z, theta, -(np.pi)/2, (np.pi)/2)(beta)
gradIntZ = T.grad(intZ, [beta])
funcIntZ = theano.function([beta], intZ)
funcGradIntZ = theano.function([beta], gradIntZ)
print (funcIntZ(2.5))
print (funcGradIntZ(2.5))



