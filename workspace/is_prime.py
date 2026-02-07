"""
Prime Number Checker Module

This module provides a function to check if a given number is prime.
"""

import math


def is_prime(number):
    """
    Check if a number is prime.
    
    A prime number is a natural number greater than 1 that has no positive
    divisors other than 1 and itself.
    
    Parameters:
    -----------
    number : int
        The integer to check for primality.
    
    Returns:
    --------
    bool
        True if the number is prime, False otherwise.
    
    Examples:
    ---------
    >>> is_prime(2)
    True
    >>> is_prime(17)
    True
    >>> is_prime(4)
    False
    >>> is_prime(1)
    False
    >>> is_prime(-5)
    False
    
    Notes:
    ------
    - Numbers less than 2 (including negative numbers, 0, and 1) are not prime.
    - The function uses an efficient algorithm that checks divisibility only up
      to the square root of the number.
    """
    # Handle edge cases
    if number < 2:
        return False
    
    # 2 is the only even prime number
    if number == 2:
        return True
    
    # All other even numbers are not prime
    if number % 2 == 0:
        return False
    
    # Check odd divisors up to the square root of the number
    # We only need to check up to sqrt(number) because if number = a * b,
    # then one of a or b must be <= sqrt(number)
    sqrt_number = int(math.sqrt(number))
    for divisor in range(3, sqrt_number + 1, 2):
        if number % divisor == 0:
            return False
    
    return True


if __name__ == "__main__":
    # Test cases demonstrating usage
    print("Prime Number Checker - Test Cases")
    print("=" * 40)
    
    # Test edge cases
    test_cases = [
        -5,    # Negative number
        0,     # Zero
        1,     # One
        2,     # Smallest prime
        3,     # Small prime
        4,     # Small composite
        17,    # Prime
        18,    # Composite
        19,    # Prime
        20,    # Composite
        97,    # Larger prime
        100,   # Larger composite
        541,   # Prime
        1000,  # Composite
        7919,  # Large prime
    ]
    
    for num in test_cases:
        result = is_prime(num)
        print(f"is_prime({num:5d}) = {result}")
    
    print("\n" + "=" * 40)
    print("Additional Examples:")
    print("=" * 40)
    
    # Find first 20 prime numbers
    print("\nFirst 20 prime numbers:")
    primes = []
    num = 2
    while len(primes) < 20:
        if is_prime(num):
            primes.append(num)
        num += 1
    print(primes)
