// Linear assignment with a cost limit, as Ultralytics' matching.linear_assignment computes it
// with lap.lapjv(cost, extend_cost=True, cost_limit=thresh): the cost matrix is extended to a
// square (rows + cols) matrix in which leaving a row or a column unmatched costs thresh / 2, so a
// pair is matched only when that lowers the total cost (globally optimal, not greedy).
//
// Solved with the Hungarian algorithm (shortest augmenting paths with potentials, O(N^3)) on that
// extended matrix, which is never built: its entries are computed from the real one. Rows and
// columns with no entry below the limit are left out first; in an optimal solution they are
// never matched, and the scenes this tracks have few objects, mostly far apart.
//
// Scratch buffers are kept between calls, so a call allocates nothing once they're large enough.

let cap = 0;
let u, v, minv, rowIdx, colIdx, p, way, used;

function ensure(n) {
  if (n <= cap) return;
  cap = Math.max(n, 2 * cap, 16);
  u = new Float64Array(cap + 1);
  v = new Float64Array(cap + 1);
  minv = new Float64Array(cap + 1);
  rowIdx = new Int32Array(cap);
  colIdx = new Int32Array(cap);
  p = new Int32Array(cap + 1);
  way = new Int32Array(cap + 1);
  used = new Uint8Array(cap + 1);
}

/**
 * `cost`: row-major nRows x nCols. Fills `rowToCol` (length >= nRows) and `colToRow`
 * (length >= nCols) with the matched index, or -1. Returns the number of matches.
 */
export function linearAssignment(cost, nRows, nCols, limit, rowToCol, colToRow) {
  rowToCol.fill(-1, 0, nRows);
  colToRow.fill(-1, 0, nCols);
  if (nRows === 0 || nCols === 0) return 0;
  ensure(nRows + nCols);

  // rows / columns with at least one entry below the limit
  let n = 0, m = 0;
  for (let i = 0; i < nRows; i++) {
    const off = i * nCols;
    for (let j = 0; j < nCols; j++) {
      if (cost[off + j] < limit) { rowIdx[n++] = i; break; }
    }
  }
  if (n === 0) return 0;
  for (let j = 0; j < nCols; j++) {
    for (let k = 0; k < n; k++) {
      if (cost[rowIdx[k] * nCols + j] < limit) { colIdx[m++] = j; break; }
    }
  }

  // Hungarian algorithm on the (n + m) square extended matrix, 1-based; p[j] = row of column j
  const size = n + m, half = limit / 2;
  u.fill(0, 0, size + 1);
  v.fill(0, 0, size + 1);
  p.fill(0, 0, size + 1);
  for (let i = 1; i <= size; i++) {
    p[0] = i;
    let j0 = 0;
    minv.fill(Infinity, 0, size + 1);
    used.fill(0, 0, size + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      const realRow = i0 <= n;
      const rowOff = realRow ? rowIdx[i0 - 1] * nCols : 0;
      const ui = u[i0];
      let delta = Infinity, j1 = 0;
      for (let j = 1; j <= size; j++) {
        if (used[j]) continue;
        // extended entry: real cost | thresh / 2 to leave a row or column unmatched | 0
        const c = realRow
          ? (j <= m ? cost[rowOff + colIdx[j - 1]] : half)
          : (j <= m ? half : 0);
        const cur = c - ui - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= size; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }

  let matches = 0;
  for (let j = 1; j <= m; j++) {
    const i = p[j];
    if (i >= 1 && i <= n) {
      const r = rowIdx[i - 1], c = colIdx[j - 1];
      rowToCol[r] = c;
      colToRow[c] = r;
      matches++;
    }
  }
  return matches;
}
