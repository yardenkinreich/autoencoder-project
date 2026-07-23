import random

def seeded_choice(seed, a, b):
    """
    Return an integer in [a, b), seeded for reproducibility,
    using the same algorithm as random.choice (rejection-sampling
    via getrandbits, as in random._randbelow).
    """
    rng = random.Random(seed)
    n = b - a
    k = n.bit_length()
    r = rng.getrandbits(k)
    while r >= n:
        r = rng.getrandbits(k)
    return a + r

if __name__ == "__main__":
    print(seeded_choice(3, 0, 15226))
