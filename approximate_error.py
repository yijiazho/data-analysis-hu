import math

def maclaurin_series(x: float, n: int) -> float:
    return ((-1) ** n) * (x ** (2 * n)) / math.factorial(2 * n)
    

def maclaurin_series_arctan(x: float, n: int) -> float:
    return ((-1) ** n) * x ** (2 * n + 1) / (2  *n + 1)
 
def main():
    x = math.pi / 4
    true_value = math.cos(x)
    print(f"\nTrue Value: {true_value}")
    print("---------------------------")
    
    n = 0
    approximation = 0
    previous_approximation = 0
    approximate_percent_error = 100  
    
    print(f"{'Term':<5} {'Approximation':<20} {'True Percent Relative Error (%)':<20} {'Approximate Percent Relative Error (%)':<20}")
    while approximate_percent_error > 1:  # smaller than 1%
        term = maclaurin_series_arctan(x, n)
        approximation += term
    
        true_percent_error = abs((true_value - approximation) / true_value) * 100
        if n > 0:  
            approximate_percent_error = abs((approximation - previous_approximation) / approximation) * 100
        
            
        print(f"{n:<5} {approximation:<20} {true_percent_error:<31} {approximate_percent_error:<25}")
        
        previous_approximation = approximation
        n += 1
    

if __name__ == "__main__":
    main()
    print(123456)
 