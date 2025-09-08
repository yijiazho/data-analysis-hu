
from typing import List


def multiply(Y: List[List[int]], Z: List[List[int]]) -> List[List[int]]:
    m = len(Y)
    n = len(Y[0])
    p = len(Z[0])
    if len(Z) != n:
        raise ValueError('Matrix size do not match')
    
    # X should be m * p
    X = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                X[i][j] += Y[i][k] * Z[k][j]
                
    return X

def print_matrix(A: List[List[int]]) -> None:
    for row in A:
        print(row)        

def main():
    A = [[1, 5], [3, 10], [-4, 3]]
    B = [[4, 3], [0.5, 2]]
    C = [[2, -2], [-3, 5]]
    print_matrix(multiply(A, B))
    
    
    
if __name__ == "__main__":
    main()
