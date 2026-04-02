/**
 * Render a compact markdown table from column headers and row data.
 *
 * Cell values are sanitised: pipes are escaped and newlines replaced with spaces.
 */
export function formatTable(
  columns: string[],
  rows: string[][],
): string {
  if (rows.length === 0) return '(no results)';

  const escape = (v: string) => v.replace(/\|/g, '\\|').replace(/\n/g, ' ');

  const header = `| ${columns.map(escape).join(' | ')} |`;
  const sep = `|${columns.map(() => '---').join('|')}|`;
  const body = rows.map(
    (cells) => `| ${cells.map(escape).join(' | ')} |`,
  );

  return [header, sep, ...body].join('\n');
}
