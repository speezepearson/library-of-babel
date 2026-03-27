import { describe, it, expect } from 'vitest';

/** Book number (0..159) is used directly as the seed. */
type BookNumber = number;

function bookNumberToSeed(bookNumber: BookNumber): bigint {
  return BigInt(bookNumber);
}

function bookPosition(bookNumber: BookNumber): { row: number; col: number } {
  const BOOKS_PER_ROW = 32;
  return {
    row: Math.floor(bookNumber / BOOKS_PER_ROW),
    col: bookNumber % BOOKS_PER_ROW,
  };
}

describe('fancy UI routing', () => {
  it('maps book numbers 0..159 to seeds', () => {
    expect(bookNumberToSeed(0)).toBe(0n);
    expect(bookNumberToSeed(42)).toBe(42n);
    expect(bookNumberToSeed(159)).toBe(159n);
  });

  it('computes correct row/col for book positions', () => {
    expect(bookPosition(0)).toEqual({ row: 0, col: 0 });
    expect(bookPosition(31)).toEqual({ row: 0, col: 31 });
    expect(bookPosition(32)).toEqual({ row: 1, col: 0 });
    expect(bookPosition(159)).toEqual({ row: 4, col: 31 });
  });

  it('has exactly 5 rows of 32 books', () => {
    const ROWS = 5;
    const COLS = 32;
    const total = ROWS * COLS;
    expect(total).toBe(160);
    // Last book
    const last = bookPosition(total - 1);
    expect(last.row).toBe(ROWS - 1);
    expect(last.col).toBe(COLS - 1);
  });
});
