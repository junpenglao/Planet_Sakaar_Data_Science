import numpy
import theano
import theano.tensor as tt
from theano.gradient import disconnected_grad as stop_grad

x = tt.dscalar('x')
y = x ** 2
gy = tt.grad(y, x)

f = theano.function([x], gy)
f(4)

numpy.allclose(f(94.2), 188.4)

fy = theano.function([x], y)
fy(4)

def magicbox(x):
    return tt.exp(x-stop_grad(x))

y2 = magicbox(x ** 2)

fy2 = theano.function([x], y2)
fy2(4)

gy2 = tt.grad(y2, x)
f2 = theano.function([x], gy2)
f2(4)
  