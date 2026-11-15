import { useEffect, useMemo, useState } from 'react'
import { api, pct } from '../api.js'
import { HBars, Legend, SERIES } from '../charts.jsx'

const PLANES = ['Sagittal', 'Coronal', 'Axial']
const PLANE_NOTE = {
  Sagittal: 'Side view (front-to-back slices). Best for the cruciate ligaments, menisci and effusion.',
  Coronal: 'Front view (side-to-side slices). Best for collateral ligaments, menisci and joint-space narrowing.',
  Axial: 'Top-down view. Best for the kneecap joint, Baker\'s cysts and synovium.',
}

// Plain-language notes and the planes a reader would check first.
const NOTES = {
  ACL: ['The main ligament that stops the shin bone sliding forward. A tear shows as a broken or blurred band in the centre of the knee.', ['Sagittal']],
  MCL: ['Ligament on the inner side of the knee. Injury shows as thickening or bright fluid around it.', ['Coronal']],
  'Medial Meniscus': ['C-shaped cartilage cushion on the inner side. A tear shows as a bright line reaching its surface.', ['Sagittal', 'Coronal']],
  'Lateral Meniscus': ['Cartilage cushion on the outer side. Tears look like bright lines through the dark wedge.', ['Sagittal', 'Coronal']],
  'Medial OA': ['Wear of the cartilage in the inner compartment: thin cartilage, bone spurs, joint-space narrowing.', ['Coronal']],
  'Lateral OA': ['Wear of the cartilage in the outer compartment.', ['Coronal']],
  'PF OA': ['Wear of the cartilage behind the kneecap (patellofemoral joint).', ['Axial', 'Sagittal']],
  Effusion: ['Extra fluid inside the joint; appears bright on these fluid-sensitive images, often above the kneecap.', ['Sagittal', 'Axial']],
  Synovitis: ['Inflamed, thickened joint lining, usually seen together with fluid.', ['Axial']],
  "Baker's": ['A fluid-filled cyst behind the knee, between the calf muscles.', ['Axial']],
  Contusion: ['Bone bruise: bright, swollen bone marrow after an impact, often with ACL tears.', ['Sagittal', 'Coronal']],
  Fracture: ['A crack in the bone, including small avulsion or stress fractures.', ['Coronal', 'Sagittal']],
}

function confidence(p, thr) {
  const m = Math.abs(p - thr)
  return m >= 0.25 ? ['High', 'ok'] : m >= 0.1 ? ['Moderate', ''] : ['Borderline', 'warn']
}

/* ---------- headline statistics for the test set ---------- */
function Stats({ runs, sample }) {
  if (!runs) return null
  const s = runs.summary
  const tiles = [
    ['Test studies', s.studies, 'radiologist-labelled, never trained on'],
    ['Accuracy', pct(s.fused_accuracy), `image + report · ${s.decisions} decisions`],
    ['Image only', pct(s.image_accuracy), 'same studies, without the report'],
    ['≥ 10 of 12 right', `${s.fused_ge_10 + 0}`, `${pct(s.fused_ge_10 / s.studies, 0)} of studies`],
    ['Median time', `${s.median_seconds.toFixed(1)}s`, 'per study on a laptop CPU'],
    ['This study', sample?.agreement != null ? `${sample.agreement}/12` : '–', 'findings matching the radiologist'],
  ]
  return (
    <div className="grid g6" style={{ marginBottom: 12 }}>
      {tiles.map(([k, v, d]) => (
        <div key={k} className="panel kpi"><div className="k">{k}</div><div className="v">{v}</div><div className="d">{d}</div></div>
      ))}
    </div>
  )
}

/* ---------- slice viewer ---------- */
function Viewer({ sample, plane, setPlane }) {
  const [k, setK] = useState(3)
  const [bright, setBright] = useState(100)
  const [contrast, setContrast] = useState(100)
  const [play, setPlay] = useState(false)
  const n = sample.slices
  useEffect(() => { setK(Math.floor(n / 2)); setPlay(false) }, [sample.id, n])
  useEffect(() => {
    if (!play) return
    const t = setInterval(() => setK((i) => (i + 1) % n), 300)
    return () => clearInterval(t)
  }, [play, n])
  useEffect(() => {
    const onKey = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) && e.target.type !== 'range') return
      if (e.key === 'ArrowRight') setK((i) => Math.min(n - 1, i + 1))
      if (e.key === 'ArrowLeft') setK((i) => Math.max(0, i - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [n])
  const name = PLANES[plane]
  const filter = { filter: `brightness(${bright}%) contrast(${contrast}%)` }
  return (
    <div className="panel">
      <div className="panel-h">
        <span>Study {sample.id}</span>
        <div className="seg">
          {PLANES.map((p, i) => (
            <button key={p} className={plane === i ? 'on' : ''} disabled={!sample.present[i]} onClick={() => setPlane(i)}>{p}</button>
          ))}
        </div>
      </div>
      <div className="panel-b">
        <div className="viewer">
          <img src={api.sliceUrl(sample.id, name, k)} alt={`${name} slice ${k + 1}`} style={filter} />
          <span className="ov">{name.toUpperCase()} · fluid-sensitive</span>
          <span className="ov r">{k + 1}/{n}</span>
        </div>
        <div className="vctl">
          <button className="iconbtn" onClick={() => setPlay(!play)}>{play ? '❚❚ Pause' : '▶ Play'}</button>
          <input type="range" min={0} max={n - 1} value={k} onChange={(e) => setK(+e.target.value)} aria-label="Slice" />
        </div>
        <div className="strip">
          {Array.from({ length: n }, (_, i) => (
            <button key={i} className={i === k ? 'on' : ''} onClick={() => setK(i)}>
              <img src={api.sliceUrl(sample.id, name, i)} alt="" loading="lazy" style={filter} />
            </button>
          ))}
        </div>
        <div className="vctl small">
          <label>Brightness <input type="range" min={50} max={200} value={bright} onChange={(e) => setBright(+e.target.value)} /></label>
          <label>Contrast <input type="range" min={50} max={250} value={contrast} onChange={(e) => setContrast(+e.target.value)} /></label>
          <button className="iconbtn" onClick={() => { setBright(100); setContrast(100) }}>Reset</button>
        </div>
        <div className="note small"><b>{name}:</b> {PLANE_NOTE[name]} <span className="muted">Use ← → keys to step through slices.</span></div>
      </div>
    </div>
  )
}

/* ---------- findings with explainable notes ---------- */
function Findings({ result, labels, info, mode, metrics, runs, onShowPlane }) {
  const [open, setOpen] = useState(null)
  const probs = result[mode]
  const thrs = (mode === 'fused' ? result.fused_thresholds : result.thresholds) || {}
  const thrOf = (l) => thrs[l] ?? 0.5
  const isPos = (l) => (result.calls ? result.calls[l] === 1 : probs[l] >= thrs[l])
  const truth = result.truth
  const agree = truth ? labels.filter((l) => isPos(l) === (truth[l] === 1)).length : null
  const positives = labels.filter(isPos)
  const prev = metrics?.dataset?.gold_prevalence || {}
  const acc = runs?.summary?.fused_per_label_accuracy || {}
  const both = result.image && result.fused

  return (
    <>
      <div className="summary">
        <div className="small muted">Model summary</div>
        <div style={{ margin: '4px 0' }}>
          {positives.length
            ? <>Flags <b>{positives.length}</b> finding{positives.length === 1 ? '' : 's'}: {positives.map((l) => <span key={l} className="chip">{l}</span>)}</>
            : <b>No abnormal findings flagged.</b>}
        </div>
        {truth && (
          <div className="small">
            Matches the radiologist on <b className={`num ${agree >= 10 ? 'ok' : ''}`}>{agree}/{labels.length}</b> findings
            <span className="muted"> · {result.precomputed ? 'precomputed' : 'inference'} {result.seconds}s on CPU</span>
          </div>
        )}
      </div>
      <div className="scroll">
        <table className="t findings">
          <thead>
            <tr><th>Finding</th><th>Probability</th><th className="n">p</th><th>Call</th>{truth && <th>Truth</th>}</tr>
          </thead>
          <tbody>
            {labels.map((l) => {
              const p = probs[l], pos = isPos(l), [conf, cc] = confidence(p, thrOf(l))
              const [note, planes] = NOTES[l] || [info[l], []]
              return [
                <tr key={l} className={`click ${open === l ? 'hl' : ''}`} onClick={() => setOpen(open === l ? null : l)} title="Click for an explanation">
                  <td>{open === l ? '▾' : '▸'} {l}</td>
                  <td style={{ width: '30%' }}>
                    <div className="bar"><i style={{ width: `${p * 100}%`, background: pos ? 'var(--s2)' : 'var(--s1)' }} />{thrs[l] != null && <b style={{ left: `${thrs[l] * 100}%` }} />}</div>
                  </td>
                  <td className="n">{pct(p, 0)}</td>
                  <td><span className={`pill ${pos ? 'pos' : 'neg'}`}>{pos ? '● Positive' : 'Negative'}</span></td>
                  {truth && (
                    <td><span className={(truth[l] === 1) === pos ? 'ok' : 'err'}>{truth[l] === 1 ? 'Positive' : 'Negative'} {(truth[l] === 1) === pos ? '✓' : '✗'}</span></td>
                  )}
                </tr>,
                open === l && (
                  <tr key={`${l}-note`} className="explain"><td colSpan={truth ? 5 : 4}>
                    <div className="note">
                      <p><b>What it is.</b> {note}</p>
                      <p><b>Why this call.</b> The model gives {pct(p, 0)}, {p >= thrOf(l) ? 'above' : 'below'} its decision line, so it is called <b>{pos ? 'positive' : 'negative'}</b>. Confidence: <b className={cc}>{conf}</b>
                        {conf === 'Borderline' && ' — close to the line, worth a careful look.'}</p>
                      {both && (
                        <p><b>Image vs report.</b> Images alone: {pct(result.image[l], 0)} → with the report: {pct(result.fused[l], 0)}
                          {Math.abs(result.fused[l] - result.image[l]) >= 0.05 ? ` (report ${result.fused[l] > result.image[l] ? 'raised' : 'lowered'} the score).` : ' (report barely changed it).'}</p>
                      )}
                      {(prev[l] != null || acc[l] != null) && (
                        <p className="muted">
                          {prev[l] != null && <>Seen in {pct(prev[l], 0)} of labelled studies. </>}
                          {acc[l] != null && <>Model is right on {pct(acc[l], 0)} of test studies for this finding.</>}
                        </p>
                      )}
                      {planes.length > 0 && onShowPlane && (
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <span className="small muted">Look at:</span>
                          {planes.map((pl) => <button key={pl} className="iconbtn" onClick={() => onShowPlane(PLANES.indexOf(pl))}>{pl} view</button>)}
                        </div>
                      )}
                    </div>
                  </td></tr>
                ),
              ]
            })}
          </tbody>
        </table>
      </div>
      <div className="small muted" style={{ padding: '6px 12px' }}>
        Bar = probability ({thrs && Object.keys(thrs).length ? 'tick = decision threshold' : 'orange = called positive'}). Click any row for a plain-language explanation.
      </div>
      {both && (
        <div className="panel-b" style={{ borderTop: '1px solid var(--grid)' }}>
          <div className="small muted" style={{ marginBottom: 4 }}>How the report changes each score</div>
          <Legend items={[['Image only', SERIES[0]], ['Image + report', SERIES[1]]]} />
          <HBars rows={labels.map((l) => ({ label: l, image: result.image[l], fused: result.fused[l] }))}
                 series={[{ key: 'image', name: 'Image only', color: SERIES[0] }, { key: 'fused', name: 'Image + report', color: SERIES[1] }]}
                 domain={[0, 1]} fmt={(v) => pct(v, 0)} labelWidth={120} rowH={22} />
        </div>
      )}
    </>
  )
}

function Upload({ onResult }) {
  const [files, setFiles] = useState({ sagittal: [], coronal: [], axial: [] })
  const [lat, setLat] = useState('R')
  const [report, setReport] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const run = async () => {
    setBusy(true); setErr('')
    const form = new FormData()
    Object.entries(files).forEach(([plane, list]) => list.forEach((f) => form.append(plane, f)))
    form.append('laterality', lat); form.append('report', report)
    try { onResult(await api.predictUpload(form)) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  const any = Object.values(files).some((l) => l.length)
  return (
    <div className="panel">
      <div className="panel-h">Upload study <span className="sub">PNG / JPG / DICOM slices</span></div>
      <div className="panel-b grid" style={{ gap: 8 }}>
        {PLANES.map((p) => (
          <label key={p} className="drop">
            <span className="small">{p} series <span className="muted">({files[p.toLowerCase()].length} files)</span></span>
            <input type="file" multiple accept=".png,.jpg,.jpeg,.dcm"
                   onChange={(e) => setFiles({ ...files, [p.toLowerCase()]: [...e.target.files] })} />
          </label>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="small">Knee</span>
          <div className="seg">
            {['R', 'L'].map((s) => <button key={s} className={lat === s ? 'on' : ''} onClick={() => setLat(s)}>{s === 'R' ? 'Right' : 'Left'}</button>)}
          </div>
        </div>
        <textarea placeholder="Optional radiology report text (enables image + report fusion)" value={report} onChange={(e) => setReport(e.target.value)} />
        <button className="btn primary" disabled={!any || busy} onClick={run}>{busy ? 'Running…' : 'Run inference'}</button>
        {err && <div className="err small">{err}</div>}
        <div className="small muted">Slices are sorted by DICOM instance number or file name; 8 are sampled evenly from the central 80% of each series.</div>
      </div>
    </div>
  )
}

export default function Inference() {
  const [samples, setSamples] = useState([])
  const [meta, setMeta] = useState(null)
  const [stats, setStats] = useState(null)
  const [sel, setSel] = useState(null)
  const [plane, setPlane] = useState(0)
  const [source, setSource] = useState('sample')
  const [useReport, setUseReport] = useState(true)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [query, setQuery] = useState('')
  const [only, setOnly] = useState('')

  useEffect(() => {
    api.samples().then((s) => { setSamples(s); setSel(s[0] || null) }).catch((e) => setErr(e.message))
    api.labels().then(setMeta).catch(() => {})
    api.results().then(setStats).catch(() => {})
  }, [])

  const run = async () => {
    setBusy(true); setErr('')
    try { setResult(await api.predictSample(sel.id, useReport)) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  useEffect(() => { if (sel) setPlane(Math.max(0, sel.present.findIndex(Boolean))) }, [sel?.id])
  useEffect(() => {
    setResult(null)
    // populate findings as soon as a study is selected
    if (source === 'sample' && sel) api.predictSample(sel.id, useReport).then(setResult).catch((e) => setErr(e.message))
  }, [sel?.id, source, useReport])

  const list = useMemo(() => samples.filter((s) =>
    s.id.toLowerCase().includes(query.trim().toLowerCase()) && (!only || s.truth[only] === 1)), [samples, query, only])

  const mode = result?.fused ? 'fused' : 'image'

  return (
    <>
      {source === 'sample' && <Stats runs={stats?.inference_runs} sample={sel} />}
      <div className="infer">
        <div className="panel">
          <div className="panel-h">
            <div className="seg">
              <button className={source === 'sample' ? 'on' : ''} onClick={() => setSource('sample')}>Test set</button>
              <button className={source === 'upload' ? 'on' : ''} onClick={() => setSource('upload')}>Upload</button>
            </div>
          </div>
          {source === 'sample' ? (
            <>
              <div className="filters">
                <input type="search" placeholder="Search study ID" value={query} onChange={(e) => setQuery(e.target.value)} />
                <select value={only} onChange={(e) => setOnly(e.target.value)} aria-label="Filter by finding">
                  <option value="">All findings</option>
                  {(meta?.labels || []).map((l) => <option key={l} value={l}>With {l}</option>)}
                </select>
                <div className="small muted">{list.length} of {samples.length} studies</div>
              </div>
              <div className="samples">
                {list.map((s) => {
                  const n = Object.values(s.truth).reduce((a, b) => a + b, 0)
                  return (
                    <button key={s.id} className={`sample ${sel?.id === s.id ? 'on' : ''}`} onClick={() => setSel(s)}>
                      <span className="mono">{s.id}</span>
                      <span className="tags">
                        {n} finding{n === 1 ? '' : 's'}
                        {s.agreement != null && <> · <b className={s.agreement >= 10 ? 'ok' : ''}>{s.agreement}/12</b></>}
                      </span>
                    </button>
                  )
                })}
                {!list.length && <div className="panel-b small muted">{err || 'No studies match.'}</div>}
              </div>
            </>
          ) : (
            <div className="panel-b small muted">Held-out studies are radiologist-labelled and were never used for training or tuning.</div>
          )}
        </div>

        <div>
          {source === 'sample' && sel ? (sel.slices ? <Viewer sample={sel} plane={plane} setPlane={setPlane} /> : (
            <div className="panel"><div className="panel-h">Study {sel.id}</div>
              <div className="panel-b small muted">Slice images are not exported for this study.</div></div>
          )) : <Upload onResult={setResult} />}
        </div>

        <div className="panel">
          <div className="panel-h">
            Findings
            {source === 'sample' && (
              <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <label className="sub" style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  <input type="checkbox" checked={useReport} onChange={(e) => setUseReport(e.target.checked)} /> fuse report
                </label>
                <button className="btn primary" disabled={!sel || busy} onClick={run}>{busy ? 'Running…' : 'Re-run'}</button>
              </span>
            )}
          </div>
          {err && <div className="panel-b err small">{err}</div>}
          {result && meta ? (
            <Findings result={result} labels={meta.labels} info={meta.info} mode={mode}
                      metrics={stats?.metrics} runs={stats?.inference_runs}
                      onShowPlane={source === 'sample' && sel?.slices ? setPlane : null} />
          ) : (
            <div className="panel-b small muted">
              {source === 'sample' ? 'Loading findings… ' : 'Upload one or more planes and run inference. '}
              The model embeds 8 slices per plane with a frozen DINOv2 encoder, pools them into a study vector,
              and scores 12 findings with a calibrated soft-voting ensemble.
            </div>
          )}
          {result?.report && (
            <div className="panel-b" style={{ borderTop: '1px solid var(--grid)' }}>
              <div className="small muted" style={{ marginBottom: 4 }}>Radiology report (input to the rule-based reader)</div>
              <div className="report">{result.report}</div>
            </div>
          )}
        </div>
      </div>
      <div className="disclaimer small muted">
        Research and teaching demo. Predictions come from a machine-learning model and are not a medical diagnosis.
      </div>
    </>
  )
}
