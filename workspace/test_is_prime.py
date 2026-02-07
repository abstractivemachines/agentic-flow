"""
Comprehensive test suite for the is_prime function.

This test suite covers edge cases, small primes, small composites,
larger primes, larger composites, and perfect squares.
"""

import pytest
from is_prime import is_prime


class TestEdgeCases:
    """Test edge cases: negative numbers, 0, 1, and 2."""
    
    def test_negative_numbers(self):
        """Negative numbers should not be prime."""
        assert is_prime(-1) == False
        assert is_prime(-5) == False
        assert is_prime(-17) == False
        assert is_prime(-100) == False
    
    def test_zero(self):
        """Zero should not be prime."""
        assert is_prime(0) == False
    
    def test_one(self):
        """One should not be prime."""
        assert is_prime(1) == False
    
    def test_two(self):
        """Two should be prime (the only even prime)."""
        assert is_prime(2) == True


class TestSmallPrimes:
    """Test small prime numbers (single digit and small two-digit)."""
    
    def test_single_digit_primes(self):
        """Test all single-digit prime numbers."""
        single_digit_primes = [2, 3, 5, 7]
        for num in single_digit_primes:
            assert is_prime(num) == True, f"{num} should be prime"
    
    def test_small_two_digit_primes(self):
        """Test small two-digit prime numbers."""
        small_primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for num in small_primes:
            assert is_prime(num) == True, f"{num} should be prime"


class TestSmallComposites:
    """Test small composite numbers."""
    
    def test_even_composites(self):
        """Even numbers greater than 2 should not be prime."""
        even_numbers = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        for num in even_numbers:
            assert is_prime(num) == False, f"{num} should not be prime"
    
    def test_odd_composites(self):
        """Odd composite numbers should not be prime."""
        odd_composites = [9, 15, 21, 25, 27, 33, 35, 39, 45, 49, 51, 55]
        for num in odd_composites:
            assert is_prime(num) == False, f"{num} should not be prime"


class TestLargerPrimes:
    """Test larger prime numbers."""
    
    def test_two_digit_primes(self):
        """Test two-digit prime numbers."""
        two_digit_primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for num in two_digit_primes:
            assert is_prime(num) == True, f"{num} should be prime"
    
    def test_three_digit_primes(self):
        """Test three-digit prime numbers."""
        three_digit_primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                              199, 211, 223, 251, 257, 263, 269, 271, 277, 281,
                              503, 509, 521, 541, 547, 557, 563, 569, 571, 577,
                              997]
        for num in three_digit_primes:
            assert is_prime(num) == True, f"{num} should be prime"
    
    def test_four_digit_primes(self):
        """Test four-digit prime numbers."""
        four_digit_primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
                             7919, 7927, 7933, 7937, 7949]
        for num in four_digit_primes:
            assert is_prime(num) == True, f"{num} should be prime"


class TestLargerComposites:
    """Test larger composite numbers."""
    
    def test_three_digit_composites(self):
        """Test three-digit composite numbers."""
        composites = [100, 102, 104, 105, 106, 108, 110, 111, 114, 115,
                      200, 300, 400, 500, 600, 700, 800, 900,
                      999]
        for num in composites:
            assert is_prime(num) == False, f"{num} should not be prime"
    
    def test_four_digit_composites(self):
        """Test four-digit composite numbers."""
        composites = [1000, 1001, 1002, 1004, 1005, 1006, 1008, 1010,
                      2000, 3000, 4000, 5000, 7920, 7921, 7922]
        for num in composites:
            assert is_prime(num) == False, f"{num} should not be prime"


class TestPerfectSquares:
    """Test perfect squares (which should never be prime except 4)."""
    
    def test_small_perfect_squares(self):
        """Test small perfect squares."""
        perfect_squares = [4, 9, 16, 25, 36, 49, 64, 81]
        for num in perfect_squares:
            assert is_prime(num) == False, f"{num} is a perfect square and should not be prime"
    
    def test_larger_perfect_squares(self):
        """Test larger perfect squares."""
        perfect_squares = [100, 121, 144, 169, 196, 225, 256, 289,
                          324, 361, 400, 441, 484, 529, 576, 625,
                          961, 1024, 1225, 1369, 1521, 1681, 1849,
                          2025, 2209, 2401, 2601, 2809, 3025]
        for num in perfect_squares:
            assert is_prime(num) == False, f"{num} is a perfect square and should not be prime"


class TestSpecificCases:
    """Test specific important or edge cases."""
    
    def test_mersenne_primes(self):
        """Test some Mersenne prime numbers (2^n - 1)."""
        mersenne_primes = [3, 7, 31, 127]  # 2^2-1, 2^3-1, 2^5-1, 2^7-1
        for num in mersenne_primes:
            assert is_prime(num) == True, f"{num} is a Mersenne prime"
    
    def test_twin_primes(self):
        """Test twin prime pairs (primes that differ by 2)."""
        twin_primes = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31),
                       (41, 43), (59, 61), (71, 73)]
        for p1, p2 in twin_primes:
            assert is_prime(p1) == True, f"{p1} should be prime"
            assert is_prime(p2) == True, f"{p2} should be prime"
    
    def test_powers_of_two(self):
        """Test powers of 2 (only 2 itself should be prime)."""
        assert is_prime(2) == True  # 2^1
        assert is_prime(4) == False  # 2^2
        assert is_prime(8) == False  # 2^3
        assert is_prime(16) == False  # 2^4
        assert is_prime(32) == False  # 2^5
        assert is_prime(64) == False  # 2^6
        assert is_prime(128) == False  # 2^7
        assert is_prime(256) == False  # 2^8
        assert is_prime(512) == False  # 2^9
        assert is_prime(1024) == False  # 2^10
    
    def test_powers_of_three(self):
        """Test powers of 3 (only 3 itself should be prime)."""
        assert is_prime(3) == True  # 3^1
        assert is_prime(9) == False  # 3^2
        assert is_prime(27) == False  # 3^3
        assert is_prime(81) == False  # 3^4
        assert is_prime(243) == False  # 3^5
        assert is_prime(729) == False  # 3^6
    
    def test_products_of_two_primes(self):
        """Test semi-primes (products of exactly two primes)."""
        semi_primes = [
            6,    # 2 * 3
            10,   # 2 * 5
            14,   # 2 * 7
            15,   # 3 * 5
            21,   # 3 * 7
            22,   # 2 * 11
            26,   # 2 * 13
            33,   # 3 * 11
            34,   # 2 * 17
            35,   # 5 * 7
            38,   # 2 * 19
            39,   # 3 * 13
            46,   # 2 * 23
            51,   # 3 * 17
            55,   # 5 * 11
            57,   # 3 * 19
            58,   # 2 * 29
        ]
        for num in semi_primes:
            assert is_prime(num) == False, f"{num} is a semi-prime and should not be prime"


class TestBoundaryValues:
    """Test boundary values and numbers near important thresholds."""
    
    def test_around_100(self):
        """Test values around 100."""
        # Primes near 100
        assert is_prime(97) == True
        assert is_prime(101) == True
        assert is_prime(103) == True
        
        # Composites near 100
        assert is_prime(98) == False
        assert is_prime(99) == False
        assert is_prime(100) == False
        assert is_prime(102) == False
    
    def test_around_1000(self):
        """Test values around 1000."""
        # Primes near 1000
        assert is_prime(997) == True
        assert is_prime(1009) == True
        
        # Composites near 1000
        assert is_prime(996) == False
        assert is_prime(998) == False
        assert is_prime(999) == False
        assert is_prime(1000) == False
        assert is_prime(1001) == False


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
