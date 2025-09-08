import math
print(15 * math.pi / 4)

def f(x):
    return 6 + 3 * math.cos(x)

def trapezoidal(f, n, a, b):
    h = (b - a) / n
    sum = 0
    x = a
    
    sum += f(x)
    for i in range(1, n):
        x += a
        sum += 2 * f(x)
    
    sum += f(b)
    return sum * h / 2

def simpson13(f, a, b, n):
    h = (b - a) / n
    sum = f(a)
    for i in range(1, n, 2):
        x = a + h * i
        sum += 4 * f(x)
        
    for j in range(2, n - 1, 2):
        x = a + h * j
        sum += 2 * f(x)
    
    sum += f(b)
    return (b - a) * (sum) / (3 * n)

def simpson38(f, a, b, n):
    h = (b - a) / n
    sum = 0
    
    for i in range(0, n - 2, 3):
        x0 = a + i * h
        x1 = x0 + h
        x2 = x1 + h
        x3 = x2 + h
        
        sum += 3 / 8 * h * (f(x0) + 3 * f(x1) + 3 * f(x2) + f(x3))
    
    return sum

print(simpson13(f, 0, math.pi / 2, 2))
print(simpson38(f, 0, math.pi / 2, 3))
print(simpson38(f, 0, math.pi / 2, 6))

# print(trapezoidal(f, 2, 0, math.pi / 2))


def g (x):
    return 1 - math.exp(-2 * x)

print(math.e)