import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { ensureAscending } from './quantize';

describe('ensureAscending', () => {
  // -- Property-based tests --------------------------------------------------

  it('output is strictly increasing', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 1, maxLength: 500 })
          .map((xs) => xs.sort((a, b) => a - b)),
        (xs) => {
          ensureAscending(xs);
          for (let i = 1; i < xs.length; i++) {
            expect(xs[i]).toBeGreaterThan(xs[i - 1]);
          }
        },
      ),
    );
  });

  it('first element is > 0', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 1, maxLength: 500 })
          .map((xs) => xs.sort((a, b) => a - b)),
        (xs) => {
          ensureAscending(xs);
          expect(xs[0]).toBeGreaterThan(0);
        },
      ),
    );
  });

  it('last element is exactly 1', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 1, maxLength: 500 })
          .map((xs) => xs.sort((a, b) => a - b)),
        (xs) => {
          ensureAscending(xs);
          expect(xs[xs.length - 1]).toBe(1);
        },
      ),
    );
  });

  // -- Specific edge cases ---------------------------------------------------

  it('handles [0,0,0,0,0,1,1,1,1,1]', () => {
    const xs = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    ensureAscending(xs);

    expect(xs[0]).toBeGreaterThan(0);
    expect(xs[xs.length - 1]).toBe(1);
    for (let i = 1; i < xs.length; i++) {
      expect(xs[i]).toBeGreaterThan(xs[i - 1]);
    }
  });

  it('handles single element', () => {
    const xs = [0.5];
    ensureAscending(xs);
    expect(xs[0]).toBe(1);
  });

  it('handles already-correct input', () => {
    const xs = [0.25, 0.5, 0.75, 1];
    const before = [...xs];
    ensureAscending(xs);
    // Values should be unchanged (or only nudged by minimal amounts)
    for (let i = 0; i < xs.length; i++) {
      expect(xs[i]).toBeCloseTo(before[i], 10);
    }
    expect(xs[3]).toBe(1);
  });

  it('handles all zeros', () => {
    const xs = [0, 0, 0, 0, 0];
    ensureAscending(xs);
    expect(xs[0]).toBeGreaterThan(0);
    expect(xs[xs.length - 1]).toBe(1);
    for (let i = 1; i < xs.length; i++) {
      expect(xs[i]).toBeGreaterThan(xs[i - 1]);
    }
  });

  it('handles all ones', () => {
    const xs = [1, 1, 1, 1, 1];
    ensureAscending(xs);
    expect(xs[0]).toBeGreaterThan(0);
    expect(xs[xs.length - 1]).toBe(1);
    for (let i = 1; i < xs.length; i++) {
      expect(xs[i]).toBeGreaterThan(xs[i - 1]);
    }
  });
});
