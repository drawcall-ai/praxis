import { describe, it, expect } from 'vitest';
import { formatTable } from '../src/format.js';

describe('formatTable', () => {
  it('renders a basic table', () => {
    const result = formatTable(
      ['#', 'name', 'score'],
      [
        ['0', 'alice', '1'],
        ['1', 'bob', '0.5'],
      ],
    );
    expect(result).toBe(
      [
        '| # | name | score |',
        '|---|---|---|',
        '| 0 | alice | 1 |',
        '| 1 | bob | 0.5 |',
      ].join('\n'),
    );
  });

  it('returns "(no results)" for empty rows', () => {
    expect(formatTable(['a', 'b'], [])).toBe('(no results)');
  });

  it('escapes pipe characters in cells', () => {
    const result = formatTable(['val'], [['a|b']]);
    expect(result).toContain('a\\|b');
    // should not produce an extra column
    const lines = result.split('\n');
    expect(lines[2]).toBe('| a\\|b |');
  });

  it('replaces newlines with spaces in cells', () => {
    const result = formatTable(['val'], [['line1\nline2']]);
    expect(result).toContain('line1 line2');
    expect(result).not.toContain('\nline2');
  });

  it('escapes pipe characters in headers', () => {
    const result = formatTable(['a|b'], [['x']]);
    expect(result.split('\n')[0]).toBe('| a\\|b |');
  });

  it('handles a single column and single row', () => {
    const result = formatTable(['x'], [['1']]);
    expect(result).toBe(
      ['| x |', '|---|', '| 1 |'].join('\n'),
    );
  });
});
