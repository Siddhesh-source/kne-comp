import { useEffect, useState } from 'react'
import { api, pct } from '../api.js'

const PLANES = ['Sagittal', 'Coronal', 'Axial']

function Viewer({ sample }) {
  const [plane, setPlane] = useState(0)
  const [k, setK] = useState(3)
  useEffect(() => { setK(3); setPlane(sample.present.findIndex(Boolean)) }, [sample.id])
  const name = PLANES[plane]
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
          <img src={api.sliceUrl(sample.id, name, k)} alt={`${name} slice ${k + 1}`} />
          <span className="ov">{name.toUpperCase()} · FS / fluid-sensitive</span>
          <span className="ov r">{k + 1}/{sample.slices}</span>
        </div>
        <input type="range" min={0} max={sample.slices - 1} value={k} onChange={(e) => setK(+e.target.value)}
               style={{ width: '100%', marginTop: 8 }} aria-label="Slice" />
        <div className="strip">
          {Array.from({ length: sample.slices }, (_, i) => (
            <button key={i} className={i === k ? 'on' : ''} onClick={() => setK(i)}>
              <img src={api.sliceUrl(sample.id, name, i)} alt="" loading="lazy" />
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function Findings({ result, labels, info, mode }) {
  const probs = result[mode]
  const thr = mode === 'fused' ? result.fused_thresholds : result.thresholds
  const truth = result.truth
  const agree = truth ? labels.filter((l) => (probs[l] >= thr[l]) === (truth[l] === 1)).length : null
  return (
    <>
      {truth && (
        <div className="small" style={{ padding: '8px 12px', borderBottom: '1px solid var(--grid)' }}>
          Agreement with radiologist labels: <b className="num">{agree}/{labels.length}</b>
          <span className="muted"> · inference {result.seconds}s on CPU</span>
        </div>
      )}
      <div className="scroll">
        <table className="t">
          <thead>
            <tr><th>Finding</th><th>Probability</th><th className="n">p</th><th>Call</th>{truth && <th>Truth</th>}</tr>
          </thead>
          <tbody>
            {labels.map((l) => {
              const p = probs[l], pos = p >= thr[l]
              return (
                <tr key={l} title={info[l]}>
                  <td>{l}</td>
                  <td style={{ width: '30%' }}>
                    <div className="bar"><i style={{ width: `${p * 100}%` }} /><b style={{ left: `${thr[l] * 100}%` }} /></div>
                  </td>
                  <td className="n">{pct(p, 0)}</td>
                  <td><span className={`pill ${pos ? 'pos' : 'neg'}`}>{pos ? '● Positive' : 'Negative'}</span></td>
                  {truth && (
                    <td>
                      <span className={(truth[l] === 1) === pos ? 'ok' : 'err'}>
                        {truth[l] === 1 ? 'Positive' : 'Negative'} {(truth[l] === 1) === pos ? '✓' : '✗'}
                      </span>
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="small muted" style={{ padding: '6px 12px' }}>Bar tick = per-finding decision threshold (selected on cross-validation).</div>
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
  const [sel, setSel] = useState(null)
  const [source, setSource] = useState('sample')
  const [useReport, setUseReport] = useState(true)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => {
    api.samples().then((s) => { setSamples(s); setSel(s[0] || null) }).catch((e) => setErr(e.message))
    api.labels().then(setMeta).catch(() => {})
  }, [])

  const run = async () => {
    setBusy(true); setErr('')
    try { setResult(await api.predictSample(sel.id, useReport)) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  useEffect(() => { setResult(null) }, [sel?.id, source])

  const mode = result?.fused ? 'fused' : 'image'

  return (
    <div className="infer">
      <div className="panel">
        <div className="panel-h">
          <div className="seg">
            <button className={source === 'sample' ? 'on' : ''} onClick={() => setSource('sample')}>Test set</button>
            <button className={source === 'upload' ? 'on' : ''} onClick={() => setSource('upload')}>Upload</button>
          </div>
        </div>
        {source === 'sample' ? (
          <div className="samples">
            {samples.map((s) => {
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
            {!samples.length && <div className="panel-b small muted">{err || 'No demo studies exported.'}</div>}
          </div>
        ) : (
          <div className="panel-b small muted">Held-out studies are radiologist-labelled and were never used for training or tuning.</div>
        )}
      </div>

      <div>
        {source === 'sample' && sel ? <Viewer sample={sel} /> : <Upload onResult={setResult} />}
      </div>

      <div className="panel">
        <div className="panel-h">
          Findings
          {source === 'sample' && (
            <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label className="sub" style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                <input type="checkbox" checked={useReport} onChange={(e) => setUseReport(e.target.checked)} /> fuse report
              </label>
              <button className="btn primary" disabled={!sel || busy} onClick={run}>{busy ? 'Running…' : 'Predict'}</button>
            </span>
          )}
        </div>
        {err && <div className="panel-b err small">{err}</div>}
        {result && meta ? (
          <Findings result={result} labels={meta.labels} info={meta.info} mode={mode} />
        ) : (
          <div className="panel-b small muted">
            {source === 'sample' ? 'Select a held-out study and press Predict. ' : 'Upload one or more planes and run inference. '}
            The model embeds 8 slices per plane with a frozen DINOv2 encoder, pools them into a study vector,
            and scores 12 findings with a calibrated soft-voting ensemble.
          </div>
        )}
        {result?.report && (
          <div className="panel-b" style={{ borderTop: '1px solid var(--grid)' }}>
            <div className="small muted" style={{ marginBottom: 4 }}>Report (rule-based reader input)</div>
            <div className="report">{result.report}</div>
          </div>
        )}
      </div>
    </div>
  )
}
