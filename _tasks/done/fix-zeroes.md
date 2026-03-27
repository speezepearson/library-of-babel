Lamentably, small ints reliably result in short outputs. Let's rethink the arithmetic coding scheme.

I need a function that:
- takes a natural number
- returns a number between 0 and 1
- has dense outputs
- doesn't especially map consecutive inputs to outputs with unusually similar prefixes (although they don't need to be farther apart than by chance)
- is relatively simple to stream arbitrarily many bits of
- doesn't get to some point and then just go 00000 or 11111 forever
- is quasi-decodable: given an interval, it's relatively simple to find a number that maps into that interval

Mathematical elegance is a strong plus but not a requirement.

I briefly considered "take the input, sha256 it to get the first 256 bits, append the input and hash again to get the next 256 bits, etc"; but sha256 isn't provably surjective, so there might be some 256-bit string that isn't a prefix of any output. ("Practical purposes" isn't enough; I need an algorithm that will *obviously with certainty* have every finite bitstring as a prefix at some point.)

I fear compounding floating point error, so I'd be most comfortable with an algorithm that doesn't handle floating point numbers at any point.


Best proposal so far:

2. Van der Corput ⊕ irrational: f(n) = VdC₂(n) XOR α (bitwise XOR of binary expansions)
VdC₂(n) reverses the binary digits of n after the point (so 5 = 101₂ maps to 0.101₂). XOR each bit with the corresponding bit of some fixed irrational α (say √2 − 1). Bit k of f(n) is just (bit k of VdC(n)) ⊕ (bit k of α) — trivially streamable with zero carry propagation. VdC(n) has finitely many nonzero bits, so the tail is α's tail — irrational, never terminates. Dense because VdC hits every k-bit prefix (across n = 0 to 2ᵏ−1), and XOR with a fixed mask is a bijection on prefixes. Quasi-decoding: to land in an interval, determine the required prefix bits, XOR with α's bits, reverse → that's n's low-order bits.



Tentative implementation:

```typescript
/**
 * Maps natural numbers to (0,1) via Van der Corput ⊕ (√2 − 1).
 *
 * Every finite bitstring appears as a prefix of some output.
 * No floating-point arithmetic is used anywhere.
 */

/**
 * Yields the binary expansion of √2 − 1 = 0.01101010…₂
 *
 * Maintains the invariant x = a√2 + b where a,b are integers.
 * Each step doubles x and emits 1 (subtracting 1) if x ≥ 1.
 * The comparison a√2 ≥ c is decided by comparing 2a² to c²,
 * which is exact because √2 is irrational (so ties never occur).
 */
function* sqrt2minus1bits(): Generator<0 | 1, never, never> {
  let a = 1n;  // x₀ = 1·√2 + (−1) = √2 − 1
  let b = -1n;
  while (true) {
    // x ← 2x
    a <<= 1n;
    b <<= 1n;
    // is x ≥ 1?  i.e. is a√2 + b ≥ 1?  i.e. is a√2 ≥ 1−b?
    const rhs = 1n - b;
    if (rhs <= 0n || 2n * a * a >= rhs * rhs) {
      yield 1;
      b -= 1n; // x ← x − 1
    } else {
      yield 0;
    }
  }
}

/**
 * Yields the bits of f(n) = VdC₂(n) ⊕ (√2 − 1), forever.
 *
 * VdC₂(n) simply reads n's bits from LSB to MSB, then 0s forever.
 * Once n's bits are exhausted the output is just √2 − 1's tail,
 * which is irrational — so the stream never terminates or degenerates.
 */
function* int2bits(n: bigint): Generator<0 | 1, never, never> {
  const α = sqrt2minus1bits();
  let remaining = n;
  while (true) {
    const vdcBit = Number(remaining & 1n) as 0 | 1;
    remaining >>= 1n;
    yield (vdcBit ^ α.next().value) as 0 | 1;
  }
}

/**
 * Given a target bit-prefix, returns an n whose image starts with those bits.
 *
 * Since f(n) bit i = (bit i of n) ⊕ (bit i of α),
 * we just XOR the target with α to recover n's bits.
 * Higher bits of n are free; setting them to 0 gives the smallest solution.
 */
function bits2int(bits: Array<0 | 1>): bigint {
  const α = sqrt2minus1bits();
  let n = 0n;
  for (let i = 0; i < bits.length; i++) {
    if (bits[i] ^ α.next().value) {
      n |= 1n << BigInt(i);
    }
  }
  return n;
}

// ── quick smoke test ────────────────────────────────────────────────

function take(gen: Generator<0 | 1>, k: number): Array<0 | 1> {
  const out: Array<0 | 1> = [];
  for (let i = 0; i < k; i++) out.push(gen.next().value as 0 | 1);
  return out;
}

// Print first 32 bits for n = 0 … 7
for (let n = 0n; n <= 7n; n++) {
  const bits = take(int2bits(n), 32);
  console.log(`f(${n}) = 0.${bits.join("")}…`);
}

// Round-trip: pick a prefix, decode it, re-encode, verify match
const target: Array<0 | 1> = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];
const decoded = bits2int(target);
const reencoded = take(int2bits(decoded), target.length);
console.log();
console.log(`target prefix : ${target.join("")}`);
console.log(`decoded n     : ${decoded}`);
console.log(`re-encoded    : ${reencoded.join("")}`);
console.log(`round-trip ok : ${target.every((b, i) => b === reencoded[i])}`);
```
