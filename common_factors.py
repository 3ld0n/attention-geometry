#!/usr/bin/env python3
"""Print the factors of each positive integer from 1 through n."""

import argparse
from math import gcd


def factors(n: int) -> list[int]:
    """Return all positive divisors of n in ascending order."""
    if n < 1:
        return []
    result: list[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            result.append(d)
            other = n // d
            if other != d:
                result.append(other)
        d += 1
    return sorted(result)


def shared_factors(a: int, b: int) -> list[int]:
    """Return positive divisors common to both a and b."""
    return factors(gcd(a, b))


def print_shared_factors(n: int) -> None:
    """Print shared factors between n and each positive integer up to n."""
    for k in range(1, n + 1):
        print(f"{k}: {shared_factors(n, k)}")


def group_by_shared_factors(n: int) -> dict[tuple[int, ...], list[int]]:
    """Group integers 1..n by their shared-factor combination with n.

    Returns a mapping from each unique factor tuple to the integers that
    share exactly that combination with n.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for k in range(1, n + 1):
        key = tuple(shared_factors(n, k))
        groups.setdefault(key, []).append(k)
    return groups


def print_grouped_shared_factors(n: int) -> None:
    """Print each unique shared-factor combination and the integers that share it."""
    for combo, integers in group_by_shared_factors(n).items():
        print(f"{list(combo)}: {integers}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the factors of each positive integer from 1 through n."
    )
    parser.add_argument(
        "n",
        type=int,
        help="upper bound (inclusive); must be a positive integer",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="print factors shared between n and each k from 1 through n",
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help="group integers by unique shared-factor combination with n",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("n must be a positive integer")

    if args.group:
        print_grouped_shared_factors(args.n)
    elif args.shared:
        print_shared_factors(args.n)
    else:
        for k in range(1, args.n + 1):
            print(f"{k}: {factors(k)}")


if __name__ == "__main__":
    main()
