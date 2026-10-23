import { createContext, useContext, useState } from 'react'

/* ---------- shared tooltip ---------- */
const TipCtx = createContext(() => {})
export function TipProvider({ children }) {
  const [tip, setTip] = useState(null)
  return (
    <TipCtx.Provider value={setTip}>
      {children}
      {tip && (
        <div className="tip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 270), top: tip.y + 14 }}>
          {tip.title && <div className="tt">{tip.title}</div>}
          {tip.rows.map(([k, v, c]) => (
            <div key={k} style={{ display: 'flex', gap: 10, justifyContent: 'space-between' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                {c && <i style={{ width: 8, height: 8, borderRadius: 2, background: c, display: 'inline-block' }} />}{k}
              </span>
              <span className="num">{v}</span>
            </div>
          ))}
        </div>
      )}
    </TipCtx.Provider>
  )
}
const useTip = () => useContext(TipCtx)
const show = (setTip, e, title, rows) => setTip({ x: e.clientX, y: e.clientY, title, rows })

export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)']

export function Legend({ items, line }) {
  return (
    <div className="legend">
      {items.map(([name, color]) => (
        <span key={name}><i className={line ? 'line' : ''} style={{ background: color }} />{name}</span>
      ))}
    </div>
  )
}

const ticks = (lo, hi, n = 5) => Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n)

/* ---------- grouped horizontal bars (one row per category) ---------- */
export function HBars({ rows, series, domain = [0, 1], fmt = (v) => v.toFixed(3), labelWidth = 150, rowH = 24, ci }) {
  const setTip = useTip()
  const W = 560, H = rows.length * rowH + 22, plotW = W - labelWidth - 46
  const x = (v) => labelWidth + ((v - domain[0]) / (domain[1] - domain[0])) * plotW
  const band = (rowH - 8) / series.length
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setTip(null)}>
      {ticks(domain[0], domain[1], 5).map((t, i) => (
        <g key={i}>
          <line className="gridline" x1={x(t)} x2={x(t)} y1={0} y2={H - 18} />
          <text x={x(t)} y={H - 4} textAnchor="middle">{t.toFixed(2)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y0 = i * rowH + 4
        return (
          <g key={r.label}
             onMouseMove={(e) => show(setTip, e, r.label, series.map((s) => [s.name, fmt(r[s.key]), s.color]).concat(
               ci && r[ci] ? [['95% CI', `${r[ci][0].toFixed(3)} – ${r[ci][1].toFixed(3)}`]] : []))}>
            <rect x={0} y={y0 - 2} width={W} height={rowH} fill="transparent" />
            <text className="lbl" x={labelWidth - 8} y={y0 + rowH / 2 - 1} textAnchor="end">{r.label}</text>
            {series.map((s, j) => {
              const v = r[s.key]
              const w = Math.max(0, x(Math.max(v, domain[0])) - labelWidth)
              const by = y0 + j * band
              return (
                <g key={s.key}>
                  <rect x={labelWidth} y={by} width={w} height={band - 2} rx={2} fill={s.color} />
                  {j === series.length - 1 && (
                    <text x={labelWidth + w + 4} y={by + band - 4} style={{ fill: 'var(--text-2)' }}>{fmt(v)}</text>
                  )}
                </g>
              )
            })}
            {ci && r[ci] && (
              <g stroke="var(--text)" strokeWidth={1.2} opacity={0.6}>
                <line x1={x(r[ci][0])} x2={x(r[ci][1])} y1={y0 + (rowH - 8) / 2} y2={y0 + (rowH - 8) / 2} />
              </g>
            )}
          </g>
        )
      })}
    </svg>
  )
}

/* ---------- multi-series line chart (shared x, one y axis) ---------- */
export function Lines({ x, series, xLabel, yDomain, logX, fmtX = (v) => String(v), fmtY = (v) => v.toFixed(3), height = 200, markers = true }) {
  const setTip = useTip()
  const W = 520, H = height, L = 46, R = 12, T = 10, B = 30
  const xs = logX ? x.map((v) => Math.log10(v)) : x
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const all = series.flatMap((s) => s.values)
  let [y0, y1] = yDomain || [Math.min(...all), Math.max(...all)]
  if (y1 - y0 < 1e-6) { y0 -= 0.01; y1 += 0.01 }
  const px = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R)
  const py = (v) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B)
  const [hover, setHover] = useState(null)
  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect()
    const mx = ((e.clientX - box.left) / box.width) * W
    let best = 0
    xs.forEach((v, i) => { if (Math.abs(px(v) - mx) < Math.abs(px(xs[best]) - mx)) best = i })
    setHover(best)
    show(setTip, e, `${xLabel} ${fmtX(x[best])}`, series.map((s) => [s.name, fmtY(s.values[best]), s.color]))
  }
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => { setHover(null); setTip(null) }}>
      {ticks(y0, y1, 4).map((t, i) => (
        <g key={i}>
          <line className="gridline" x1={L} x2={W - R} y1={py(t)} y2={py(t)} />
          <text x={L - 6} y={py(t) + 3} textAnchor="end">{fmtY(t)}</text>
        </g>
      ))}
      {x.map((v, i) => (
        (x.length <= 12 || i % Math.ceil(x.length / 8) === 0) &&
        <text key={i} x={px(xs[i])} y={H - B + 14} textAnchor="middle">{fmtX(v)}</text>
      ))}
      <text className="lbl" x={(L + W - R) / 2} y={H - 2} textAnchor="middle">{xLabel}</text>
      {hover != null && <line x1={px(xs[hover])} x2={px(xs[hover])} y1={T} y2={H - B} stroke="var(--text-3)" strokeDasharray="3 3" />}
      {series.map((s) => (
        <g key={s.name}>
          <polyline fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round"
                    strokeDasharray={s.dash ? '5 4' : undefined}
                    points={s.values.map((v, i) => `${px(xs[i])},${py(v)}`).join(' ')} />
          {markers && s.values.map((v, i) => (
            <circle key={i} cx={px(xs[i])} cy={py(v)} r={hover === i ? 4.5 : 3} fill={s.color} stroke="var(--surface)" strokeWidth={1.5} />
          ))}
        </g>
      ))}
    </svg>
  )
}

/* ---------- heatmap (sequential single hue) ---------- */
function mix(t) {
  // interpolate surface-2 -> seq-1 via color-mix so it follows the theme
  return `color-mix(in oklab, var(--seq-1) ${Math.round(Math.max(0, Math.min(1, t)) * 100)}%, var(--seq-0))`
}
export function Heatmap({ rows, cols, values, domain, fmt = (v) => v.toFixed(2), cell = 34, labelWidth = 120, colLabelH = 70, title = (r, c) => `${r} × ${c}` }) {
  const setTip = useTip()
  const flat = values.flat()
  const [lo, hi] = domain || [Math.min(...flat), Math.max(...flat)]
  const W = labelWidth + cols.length * cell, H = colLabelH + rows.length * cell
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W * 1.3 }} onMouseLeave={() => setTip(null)}>
      {cols.map((c, j) => (
        <text key={c} className="lbl" transform={`translate(${labelWidth + j * cell + cell / 2 + 3},${colLabelH - 6}) rotate(-50)`}>{c}</text>
      ))}
      {rows.map((r, i) => (
        <g key={r}>
          <text className="lbl" x={labelWidth - 6} y={colLabelH + i * cell + cell / 2 + 4} textAnchor="end">{r}</text>
          {cols.map((c, j) => {
            const v = values[i][j]
            const t = (v - lo) / (hi - lo || 1)
            return (
              <g key={c} onMouseMove={(e) => show(setTip, e, title(r, c), [['value', fmt(v)]])}>
                <rect x={labelWidth + j * cell + 1} y={colLabelH + i * cell + 1} width={cell - 2} height={cell - 2} rx={2} fill={mix(t)} />
                <text x={labelWidth + j * cell + cell / 2} y={colLabelH + i * cell + cell / 2 + 3} textAnchor="middle"
                      style={{ fill: t > 0.55 ? '#fff' : 'var(--text-2)', fontSize: 9.5 }}>{fmt(v)}</text>
              </g>
            )
          })}
        </g>
      ))}
    </svg>
  )
}

/* ---------- 2-class scatter ---------- */
export function Scatter({ x, y, cls, names = ['Negative', 'Positive'] }) {
  const setTip = useTip()
  const W = 520, H = 300, P = 28
  const [x0, x1] = [Math.min(...x), Math.max(...x)], [y0, y1] = [Math.min(...y), Math.max(...y)]
  const px = (v) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P)
  const py = (v) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P)
  const order = x.map((_, i) => i).sort((a, b) => cls[a] - cls[b])
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setTip(null)}>
      <rect x={P} y={P / 2} width={W - 2 * P} height={H - 1.5 * P} fill="none" className="axis" />
      <text className="lbl" x={W / 2} y={H - 4} textAnchor="middle">SVD component 1</text>
      <text className="lbl" transform={`translate(12,${H / 2}) rotate(-90)`} textAnchor="middle">SVD component 2</text>
      {order.map((i) => (
        <circle key={i} cx={px(x[i])} cy={py(y[i])} r={cls[i] ? 3.2 : 2.6}
                fill={cls[i] ? 'var(--s2)' : 'var(--s1)'} fillOpacity={cls[i] ? 0.85 : 0.35}
                stroke="var(--surface)" strokeWidth={0.6}
                onMouseMove={(e) => show(setTip, e, names[cls[i]], [['SV1', x[i].toFixed(2)], ['SV2', y[i].toFixed(2)]])} />
      ))}
    </svg>
  )
}

/* ---------- small-multiple ROC ---------- */
export function MiniRoc({ title, auc, curves }) {
  const setTip = useTip()
  const S = 120, P = 6
  const p = (v) => P + v * (S - 2 * P)
  return (
    <div>
      <div className="small" style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span>{title}</span><span className="num">{auc}</span>
      </div>
      <svg className="chart" viewBox={`0 0 ${S} ${S}`} onMouseLeave={() => setTip(null)}
           onMouseMove={(e) => show(setTip, e, title, curves.map((c) => [c.name, c.auc, c.color]))}>
        <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} fill="var(--surface-2)" rx={2} />
        <line x1={p(0)} y1={p(1)} x2={p(1)} y2={p(0)} stroke="var(--text-3)" strokeDasharray="2 3" />
        {curves.map((c) => (
          <polyline key={c.name} fill="none" stroke={c.color} strokeWidth={2}
                    points={c.fpr.map((f, i) => `${p(f)},${p(1 - c.tpr[i])}`).join(' ')} />
        ))}
      </svg>
    </div>
  )
}

/* ---------- 2x2 confusion matrix ---------- */
export function Confusion({ title, m }) {
  const [[tn, fp], [fn, tp]] = m
  const max = Math.max(tn, fp, fn, tp)
  const cell = (v, label, good) => (
    <div title={label} style={{ background: mix(v / max), color: v / max > 0.55 ? '#fff' : 'var(--text-2)', fontWeight: good ? 500 : 400 }}>{v}</div>
  )
  return (
    <div>
      <div className="small" style={{ marginBottom: 3 }}>{title}</div>
      <div className="cm">
        {cell(tn, 'True negative', true)}{cell(fp, 'False positive')}
        {cell(fn, 'False negative')}{cell(tp, 'True positive', true)}
      </div>
    </div>
  )
}
