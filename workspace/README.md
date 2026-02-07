# Prime Number Checker

A Python implementation of an efficient prime number checking function.

## Features

- **Function name**: `is_prime(number)`
- **Input**: An integer number
- **Output**: Boolean (`True` if prime, `False` otherwise)
- **Efficient algorithm**: Checks divisibility only up to the square root of the number
- **Comprehensive edge case handling**:
  - Negative numbers → `False`
  - Zero → `False`
  - One → `False`
  - Two → `True` (the only even prime)
  - Even numbers > 2 → `False`

## Algorithm Efficiency

The function uses an optimized approach:
1. Handles edge cases first (numbers < 2, even numbers)
2. Only checks odd divisors from 3 to √n
3. Time complexity: O(√n)
4. Space complexity: O(1)

## Usage

```python
from is_prime import is_prime

# Check if a number is prime
print(is_prime(17))   # True
print(is_prime(18))   # False
print(is_prime(2))    # True
print(is_prime(1))    # False
print(is_prime(-5))   # False
```

## Running Tests

Run the script directly to see test cases:
```bash
python is_prime.py
```

Run doctests:
```bash
python -m doctest is_prime.py -v
```

## Example Output

The test cases demonstrate:
- Edge cases (negative, 0, 1, 2)
- Small primes and composites
- Larger numbers
- Finding the first 20 prime numbers
