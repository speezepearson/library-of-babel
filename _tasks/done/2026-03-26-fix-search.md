The prefix search is broken. TDD it:
- write a test that computes a seed for prefix `The true meaning of life is` (currently it gives me 2397594), decodes 50 tokens from that seed, and asserts that the final string begins with `The true meaning of life is`
- figure out why it's not working (or, if your test passes, figure out why it's reliably failing for me in the UI)
- green it

## Solution

**Root cause:** The bijection `n → binary(n) ++ [1] ++ zeros` can only produce
bit streams starting with `1` (for n>0), because binary representations never
have leading zeros. When the arithmetic encoder outputs bits starting with `0`
(which happens whenever the encoded interval falls below 0.5), `bitsToSmallestBigInt`
silently drops the leading zero, producing a seed whose BitStream diverges at
bit 0.

**Fix:** Changed the bijection to `n → drop_leading_1(binary(n+1)) ++ [1] ++ zeros`.
This maps N onto all dyadic rationals with odd numerator in (0,1) — covering
both halves of the unit interval. Updated both `BitStream` constructor and
`bitsToSmallestBigInt` inverse. The old bijection only covered {1/4} ∪ [1/2, 1);
the new one covers all of (0,1) densely.

Files changed: `src/lib/arithmetic.ts` (BitStream, bitsToSmallestBigInt)
Test added: "prefix search roundtrips" in `src/__tests__/smoke.test.ts`
