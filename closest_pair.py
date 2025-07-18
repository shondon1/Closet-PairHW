import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# Point class to represent a 2D point
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"({self.x}, {self.y})"

# Calculate Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ALG1: Brute Force Algorithm - O(n^2)
def BruteForceClosestPoints(P):
    """
    Check all pairs of points to find the closest pair
    Time Complexity: O(n^2)
    """
    n = len(P)
    min_dist = float('inf')
    closest_pair = (None, None)
    
    # Check every pair of points
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(P[i], P[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (i, j)
    
    return closest_pair, min_dist

# Merge Sort for sorting points - O(n log n)
def merge_sort(arr, key_func):
    """
    Merge sort implementation for sorting points
    key_func: function to extract comparison key (x or y coordinate)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key_func)
    right = merge_sort(arr[mid:], key_func)
    
    return merge(left, right, key_func)

def merge(left, right, key_func):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if key_func(left[i]) <= key_func(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Helper function for divide and conquer
def closest_in_strip(strip, d):
    """
    Find closest points in the strip around the dividing line
    """
    min_dist = d
    closest_pair = (None, None)
    
    # Sort strip by y-coordinate
    strip = merge_sort(strip, lambda p: p.y)
    
    # Check only points within distance d
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:
            dist = distance(strip[i], strip[j])
            if dist < min_dist:
                min_dist = dist
                # Find original indices
                idx1 = next(k for k, p in enumerate(P_global) if p.x == strip[i].x and p.y == strip[i].y)
                idx2 = next(k for k, p in enumerate(P_global) if p.x == strip[j].x and p.y == strip[j].y)
                closest_pair = (idx1, idx2)
            j += 1
    
    return closest_pair, min_dist

# Global variable to store original points array for index tracking
P_global = []

# ALG2: Divide and Conquer Algorithm - O(n log n)
def DivideAndConquerClosestPoints(P):
    """
    Divide and conquer approach to find closest pair
    Time Complexity: O(n log n)
    """
    global P_global
    P_global = P.copy()  # Store original array for index tracking
    
    # Sort by x-coordinate
    Px = merge_sort(P, lambda p: p.x)
    
    def divide_conquer_recursive(Px):
        n = len(Px)
        
        # Base case: use brute force for small n
        if n <= 3:
            min_dist = float('inf')
            closest_pair = (None, None)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(Px[i], Px[j])
                    if dist < min_dist:
                        min_dist = dist
                        # Find original indices
                        idx1 = next(k for k, p in enumerate(P_global) if p.x == Px[i].x and p.y == Px[i].y)
                        idx2 = next(k for k, p in enumerate(P_global) if p.x == Px[j].x and p.y == Px[j].y)
                        closest_pair = (idx1, idx2)
            return closest_pair, min_dist
        
        # Divide
        mid = n // 2
        midpoint = Px[mid]
        
        left_points = Px[:mid]
        right_points = Px[mid:]
        
        # Conquer
        left_pair, left_dist = divide_conquer_recursive(left_points)
        right_pair, right_dist = divide_conquer_recursive(right_points)
        
        # Find minimum of the two
        if left_dist < right_dist:
            min_dist = left_dist
            closest_pair = left_pair
        else:
            min_dist = right_dist
            closest_pair = right_pair
        
        # Build strip of points within min_dist of dividing line
        strip = []
        for point in Px:
            if abs(point.x - midpoint.x) < min_dist:
                strip.append(point)
        
        # Find closest points in strip
        strip_pair, strip_dist = closest_in_strip(strip, min_dist)
        
        # Return the minimum
        if strip_dist < min_dist:
            return strip_pair, strip_dist
        else:
            return closest_pair, min_dist
    
    return divide_conquer_recursive(Px)

# Generate random distinct points
def generate_random_points(n, max_coord=1000000):
    """
    Generate n distinct random points
    """
    points_set = set()
    points = []
    
    while len(points) < n:
        x = random.randint(0, max_coord)
        y = random.randint(0, max_coord)
        if (x, y) not in points_set:
            points_set.add((x, y))
            points.append(Point(x, y))
    
    return points

# Main experimental function
def run_experiments():
    """
    Run experiments as specified in the project
    """
    # Configuration
    m = 10  # number of iterations
    sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    
    # If 100k takes too long, use smaller sizes:
    # sizes = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
    
    # Results storage
    results = {
        'n': sizes,
        'alg1_times': [],
        'alg2_times': [],
        'alg1_theoretical': [],
        'alg2_theoretical': []
    }
    
    print("Starting experiments...")
    
    for n in sizes:
        print(f"\nTesting with n = {n}")
        alg1_times = []
        alg2_times = []
        
        for j in range(m):
            print(f"  Iteration {j+1}/{m}", end='\r')
            
            # Generate random points
            points = generate_random_points(n)
            
            # Test ALG1 (Brute Force)
            start_time = time.time()
            pair1, dist1 = BruteForceClosestPoints(points)
            end_time = time.time()
            alg1_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Test ALG2 (Divide and Conquer)
            start_time = time.time()
            pair2, dist2 = DivideAndConquerClosestPoints(points)
            end_time = time.time()
            alg2_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Verify both algorithms found the same distance
            if abs(dist1 - dist2) > 0.0001:
                print(f"\nWarning: Algorithms found different distances! {dist1} vs {dist2}")
        
        # Calculate average times
        avg_alg1 = sum(alg1_times) / m
        avg_alg2 = sum(alg2_times) / m
        
        results['alg1_times'].append(avg_alg1)
        results['alg2_times'].append(avg_alg2)
        results['alg1_theoretical'].append(n * n)
        results['alg2_theoretical'].append(n * math.log2(n))
        
        print(f"\n  ALG1 average: {avg_alg1:.2f} ms")
        print(f"  ALG2 average: {avg_alg2:.2f} ms")
    
    return results

# Calculate constants and create tables
def create_tables(results):
    """
    Create the tables required for the report
    """
    print("\n\nTable ALG1 (Brute Force)")
    print("-" * 80)
    print(f"{'n':>10} {'Theoretical RT':>15} {'Empirical RT':>15} {'Ratio':>20} {'Predicted RT':>15}")
    print(f"{'':>10} {'(n^2)':>15} {'(msec)':>15} {'(x10^-8)':>20} {'(msec)':>15}")
    print("-" * 80)
    
    ratios_alg1 = []
    for i, n in enumerate(results['n']):
        theoretical = results['alg1_theoretical'][i]
        empirical = results['alg1_times'][i]
        ratio = empirical / theoretical
        ratios_alg1.append(ratio)
        print(f"{n:>10} {theoretical:>15} {empirical:>15.2f} {ratio*1e8:>20.2f} ", end="")
    
    # Calculate c1 (excluding outliers if needed)
    c1 = max(ratios_alg1)
    
    # Print predicted RT
    print("\n")
    for i, n in enumerate(results['n']):
        predicted = c1 * results['alg1_theoretical'][i]
        print(f"{'':>10} {'':>15} {'':>15} {'':>20} {predicted:>15.2f}")
    
    print(f"\nc1 = {c1:.2e}")
    
    print("\n\nTable ALG2 (Divide and Conquer)")
    print("-" * 80)
    print(f"{'n':>10} {'Theoretical RT':>15} {'Empirical RT':>15} {'Ratio':>20} {'Predicted RT':>15}")
    print(f"{'':>10} {'(n*log n)':>15} {'(msec)':>15} {'(x10^-6)':>20} {'(msec)':>15}")
    print("-" * 80)
    
    ratios_alg2 = []
    for i, n in enumerate(results['n']):
        theoretical = results['alg2_theoretical'][i]
        empirical = results['alg2_times'][i]
        ratio = empirical / theoretical
        ratios_alg2.append(ratio)
        print(f"{n:>10} {theoretical:>15.2f} {empirical:>15.2f} {ratio*1e6:>20.2f} ", end="")
    
    # Calculate c2
    c2 = max(ratios_alg2)
    
    # Print predicted RT
    print("\n")
    for i, n in enumerate(results['n']):
        predicted = c2 * results['alg2_theoretical'][i]
        print(f"{'':>10} {'':>15} {'':>15} {'':>20} {predicted:>15.2f}")
    
    print(f"\nc2 = {c2:.2e}")
    
    return c1, c2

# Create graphs
def create_graphs(results, c1, c2):
    """
    Create the three required graphs
    """
    n_values = results['n']
    
    # Graph 1: Empirical RT Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, results['alg1_times'], 'o-', label='ALG1 Empirical', markersize=8)
    plt.plot(n_values, results['alg2_times'], 's-', label='ALG2 Empirical', markersize=8)
    plt.xlabel('n')
    plt.ylabel('RT (msec)')
    plt.title('Graph 1: ALG1 Empirical RT and ALG2 Empirical RT')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph1_empirical_comparison.png')
    plt.show()
    
    # Graph 2: ALG1 Empirical vs Predicted
    plt.figure(figsize=(10, 6))
    predicted_alg1 = [c1 * t for t in results['alg1_theoretical']]
    plt.plot(n_values, results['alg1_times'], 'o-', label='ALG1 Empirical RT', markersize=8)
    plt.plot(n_values, predicted_alg1, '^-', label='ALG1 Predicted RT', markersize=8)
    plt.xlabel('n')
    plt.ylabel('RT (msec)')
    plt.title('Graph 2: ALG1 Empirical RT and ALG1 Predicted RT')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph2_alg1_comparison.png')
    plt.show()
    
    # Graph 3: ALG2 Empirical vs Predicted
    plt.figure(figsize=(10, 6))
    predicted_alg2 = [c2 * t for t in results['alg2_theoretical']]
    plt.plot(n_values, results['alg2_times'], 's-', label='ALG2 Empirical RT', markersize=8)
    plt.plot(n_values, predicted_alg2, 'd-', label='ALG2 Predicted RT', markersize=8)
    plt.xlabel('n')
    plt.ylabel('RT (msec)')
    plt.title('Graph 3: ALG2 Empirical RT and ALG2 Predicted RT')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph3_alg2_comparison.png')
    plt.show()

# Test with small example
def test_algorithms():
    """
    Test both algorithms with a small example to verify correctness
    """
    print("Testing algorithms with small example...")
    
    # Create test points
    test_points = [
        Point(0, 0),
        Point(1, 1),
        Point(5, 5),
        Point(1, 2),  # Closest to (1,1)
        Point(10, 10)
    ]
    
    print("Test points:", test_points)
    
    # Test both algorithms
    pair1, dist1 = BruteForceClosestPoints(test_points)
    pair2, dist2 = DivideAndConquerClosestPoints(test_points)
    
    print(f"\nBrute Force: points {pair1} with distance {dist1:.4f}")
    print(f"Divide & Conquer: points {pair2} with distance {dist2:.4f}")
    
    if abs(dist1 - dist2) < 0.0001:
        print("✓ Both algorithms found the same distance!")
    else:
        print("✗ Algorithms found different distances!")

# Main function
def main():
    print("Closest Pair of Points - Algorithm Comparison")
    print("=" * 50)
    
    # Test algorithms first
    test_algorithms()
    
    # Run main experiments
    print("\n\nStarting main experiments...")
    results = run_experiments()
    
    # Create tables and calculate constants
    c1, c2 = create_tables(results)
    
    # Create graphs
    create_graphs(results, c1, c2)
    
    print("\nExperiments completed!")
    print("Graphs saved as: graph1_empirical_comparison.png, graph2_alg1_comparison.png, graph3_alg2_comparison.png")

if __name__ == "__main__":
    main()