from tinyec_seeded import ec, registry
from random import choice
import numpy as np
import warnings
from math import log2
from Crypto.Util import number

curve = registry.get_curve('brainpoolP256r1')
print(curve)
print(type(curve.field))
print(curve.field.g, type(curve.field.g), log2(curve.field.g[0]), log2(curve.field.g[1]))
print(curve.field.h)
print(curve.field.n, log2(curve.field.n))
print(curve.field.p, log2(curve.field.p))
print("Done")

try:
    LONG_TYPE = long
except NameError:
    LONG_TYPE = int

def egcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y


def mod_inv(a, p):
    if a < 0:
        return p - mod_inv(-a, p)
    g, x, y = egcd(a, p)
    if g != 1:
        raise ArithmeticError("Modular inverse does not exist")
    else:
        return x % p


class Curve(object):
    def __init__(self, a, b, field, name="undefined"):
        self.name = name
        self.a = a
        self.b = b
        self.field = field
        self.g = Point(self, self.field.g[0], self.field.g[1])

    def is_singular(self):
        return (4 * self.a**3 + 27 * self.b**2) % self.field.p == 0

    def on_curve(self, x, y):
        return (y**2 - x**3 - self.a * x - self.b) % self.field.p == 0

    def __eq__(self, other):
        if not isinstance(other, Curve):
            return False
        return self.a == other.a and self.b == other.b and self.field == other.field

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "\"%s\" => y^2 = x^3 + %dx + %d (mod %d)" % (self.name, self.a, self.b, self.field.p)


class SubGroup(object):
    def __init__(self, p, g, n, h):
        self.p = p
        self.g = g
        self.n = n
        self.h = h

    def __eq__(self, other):
        if not isinstance(other, SubGroup):
            return False
        return self.p == other.p and self.g == other.g and self.n == other.n and self.h == other.h

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "Subgroup => generator %s, order: %d, cofactor: %d on Field => prime %d" % (self.g, self.n,
                                                                                           self.h, self.p)

    def __repr__(self):
        return self.__str__()


class Inf(object):
    def __init__(self, curve, x=None, y=None):
        self.x = x
        self.y = y
        self.curve = curve

    def __eq__(self, other):
        if not isinstance(other, Inf):
            return False
        return self.curve == other.curve

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, Inf):
            return Inf()
        if isinstance(other, Point):
            return other
        raise TypeError("Unsupported operand type(s) for +: '%s' and '%s'" % (other.__class__.__name__,
                                                                                  self.__class__.__name__))

    def __sub__(self, other):
        if isinstance(other, Inf):
            return Inf()
        if isinstance(other, Point):
            return other
        raise TypeError("Unsupported operand type(s) for +: '%s' and '%s'" % (other.__class__.__name__,
                                                                                  self.__class__.__name__))

    def __str__(self):
        return "%s on %s" % (self.__class__.__name__, self.curve)

    def __repr__(self):
        return self.__str__()


class Point(object):
    def __init__(self, curve, x, y):
        self.curve = curve
        self.x = x
        self.y = y
        self.p = self.curve.field.p
        self.on_curve = True
        if not self.curve.on_curve(self.x, self.y):
            warnings.warn("Point (%d, %d) is not on curve \"%s\"" % (self.x, self.y, self.curve))
            self.on_curve = False

    def __m(self, p, q):
        if p.x == q.x:
            return (3 * p.x**2 + self.curve.a) * mod_inv(2 * p.y, self.p)
        else:
            return (p.y - q.y) * mod_inv(p.x - q.x, self.p)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y and self.curve == other.curve

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, Inf):
            return self
        if isinstance(other, Point):
            if self.x == other.x and self.y != other.y:
                return Inf(self.curve)
            elif self.curve == other.curve:
                m = self.__m(self, other)
                x_r = (m**2 - self.x - other.x) % self.p
                y_r = -(self.y + m * (x_r - self.x)) % self.p
                return Point(self.curve, x_r, y_r)
            else:
                raise ValueError("Cannot add points belonging to different curves")
        else:
            raise TypeError("Unsupported operand type(s) for +: '%s' and '%s'" % (other.__class__.__name__,
                                                                                  self.__class__.__name__))

    def __sub__(self, other):
        if isinstance(other, Inf):
            return self.__add__(other)
        if isinstance(other, Point):
            return self.__add__(Point(self.curve, other.x, -other.y % self.p))
        else:
            raise TypeError("Unsupported operand type(s) for -: '%s' and '%s'" % (other.__class__.__name__,
                                                                                  self.__class__.__name__))

    def __mul__(self, other):
        if isinstance(other, Inf):
            return Inf(self.curve)
        if isinstance(other, int) or isinstance(other, LONG_TYPE):
            if other % self.curve.field.n == 0:
                return Inf(self.curve)
            if other < 0:
                addend = Point(self.curve, self.x, -self.y % self.p)
            else:
                addend = self
            result = Inf(self.curve)
            # Iterate over all bits starting by the LSB
            for bit in reversed([int(i) for i in bin(abs(other))[2:]]):
                if bit == 1:
                    result += addend
                addend += addend
            return result
        else:
            raise TypeError("Unsupported operand type(s) for *: '%s' and '%s'" % (other.__class__.__name__,
                                                                                  self.__class__.__name__))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return "(%d, %d) %s %s" % (self.x, self.y, "on" if self.on_curve else "off", self.curve)

    def __repr__(self):
        return self.__str__()


def getPrime(N):
    """
    Remade from pycrypto to work with seeding.
    """
    if N < 2:
        raise ValueError("N must be larger than 1")

    while True:
        number_make = generate_random_number_unsigned(N)
        if number.isPrime(number_make):
            break
    return number_make

def seed_large_number(number_in):
    seeds = []
    if number_in < 0:
        number_in = -number_in
        seeds.append(1)
    constant = 2**32
    while number_in > 0:
        seeds.append(number_in % constant)
        # number_in -= seeds[-1]
        number_in = number_in >> 32
    np.random.seed(seeds)

def generate_public_private_key():
    public_seed = generate_random_number_unsigned(bits=512)
    private_seed = generate_random_number_unsigned(bits=512)

    seed_large_number(public_seed)
    p = getPrime(256)
    h = 1
    g_1 = generate_random_number_unsigned(bits=256) % p
    g_2 = generate_random_number_unsigned(bits=256) % p
    n = generate_random_number_unsigned(bits=256)
    g = (g_1, g_2)

    subgroup = SubGroup(p, g, n, h)
    a = generate_random_number_unsigned(bits=256)
    b = (g_2**2 - g_1**3 - a*g_1) % p
    curve = Curve(a, b, subgroup, name="Private curve")
    print(curve)

    # g = 

    seed_large_number(public_seed)
    print(log2(getPrime(256)))
    # g_1 = generate_random_number_unsigned







def generate_random_number_unsigned(bits: int = 256, signed: bool = False):
    bits_generate = np.random.choice(2, bits)
    number_generate, power = 0, 1
    for coeff in bits_generate:
        number_generate += int(coeff*power)
        power *= 2
    return int(number_generate)

print(generate_random_number_unsigned(bits=32))
generate_public_private_key()