import { useEffect, useMemo, useState } from 'react'
import { api, f3, pct } from '../api.js'
import { Confusion, HBars, Heatmap, Legend, Lines, MiniRoc, SERIES, Scatter } from '../charts.jsx'

function Panel({ title, sub, children, style }) {
  return (
    <div className="panel" style={style}>
      <div className="panel-h">{title}{sub && <span className="sub">{sub}</span>}</div>
      <div className="panel-b">{children}</div>
    </div>
  )
}

function Kpi({ k, v, d }) {
  return <div className="panel kpi"><div className="k">{k}</div><div className="v">{v}</div>{d && <div className="d">{d}</div>}</div>
}

const METRIC_COLS = [['auc', 'AUC'], ['accuracy', 'Acc'], ['precision', 'Prec'], ['recall', 'Recall'], ['specificity', 'Spec'], ['npv', 'NPV'], ['f1', 'F1']]

function LabelTable({ data, labels }) {
  return (
    <div className="scroll">
      <table className="t">
        <thead>
          <tr><th>Finding</th><th className="n">Pos</th>{METRIC_COLS.map(([, n]) => <th key={n} className="n">{n}</th>)}<th className="n">Thr</th></tr>
        </thead>
        <tbody>
          {labels.map((l) => {
            const m = data.per_label[l]
            return (
              <tr key={l}>
                <td>{l}</td><td className="n">{m.positives}</td>
                {METRIC_COLS.map(([k]) => (
                  <td key={k} className="n" style={{ background: m[k] != null ? `color-mix(in oklab, var(--seq-1) ${Math.round(Math.max(0, m[k] - 0.5) * 2 * 45)}%, transparent)` : undefined }}>
                    {f3(m[k])}
                  </td>
                ))}
                <td className="n">{m.threshold.toFixed(2)}</td>
              </tr>
            )
          })}
          <tr className="hl">
            <td><b>Macro mean</b></td><td />
            {METRIC_COLS.map(([k]) => <td key={k} className="n"><b>{f3(data.macro[k])}</b></td>)}<td />
          </tr>
        </tbody>
      </table>
    </div>
  )
}

const STRONG_KEYS = [['auc', 'ROC-AUC'], ['npv', 'NPV'], ['specificity', 'Specificity'], ['accuracy', 'Accuracy'], ['recall', 'Recall'], ['precision', 'Precision'], ['f1', 'F1']]

function Strongest({ m, runs, labels }) {
  const items = []
  for (const [mode, block] of [['Image + report', m.fusion.test], ['Image only', m.final_model.test]]) {
    for (const l of labels) {
      for (const [k, name] of STRONG_KEYS) {
        const v = block.per_label[l][k]
        if (v != null && v >= 0.85) items.push({ l, name, mode, v, pctMetric: k !== 'auc' })
      }
    }
  }
  if (runs) {
    for (const l of labels) {
      const v = runs.summary.fused_per_label_accuracy[l]
      if (v >= 0.85) items.push({ l, name: 'Web accuracy', mode: 'Image + report', v, pctMetric: true })
    }
  }
  items.sort((a, b) => b.v - a.v)
  const shown = items.slice(0, 24)
  return (
    <Panel title="Strongest results" sub={`${items.length} per-finding metrics ≥ 0.85 on the held-out test set · top ${shown.length} shown`}>
      <div className="strong">
        {shown.map((it, i) => (
          <div key={i} className="strong-i">
            <div className="v">{it.pctMetric ? pct(it.v) : f3(it.v)}</div>
            <div className="l">{it.l} · {it.name}</div>
            <div className="mode">{it.mode}</div>
          </div>
        ))}
      </div>
      <div className="small muted" style={{ marginTop: 8 }}>
        Measured per finding on the 58 radiologist-labelled studies; "Web accuracy" is from running the live web pipeline on every study.
        With few positives per finding these values carry wide uncertainty — see the per-finding table below.
      </div>
    </Panel>
  )
}

function InferenceRuns({ runs }) {
  const s = runs.summary
  const hist = Object.entries(s.fused_histogram).filter(([, c]) => c > 0).sort((a, b) => b[0] - a[0])
  const sorted = [...runs.runs].sort((a, b) => b.fused_correct - a.fused_correct || a.id.localeCompare(b.id))
  return (
    <>
      <h2 className="section">End-to-end inference on held-out studies</h2>
      <div className="grid g6">
        <Kpi k="Web accuracy" v={pct(s.fused_accuracy)} d={`image + report · ${s.decisions} decisions`} />
        <Kpi k="Image-only" v={pct(s.image_accuracy)} d="same studies, no report" />
        <Kpi k="Perfect 12/12" v={s.fused_perfect} d={`of ${s.studies} studies`} />
        <Kpi k="≥ 11 of 12" v={s.fused_ge_11} d={`${pct(s.fused_ge_11 / s.studies, 0)} of studies`} />
        <Kpi k="≥ 10 of 12" v={s.fused_ge_10} d={`${pct(s.fused_ge_10 / s.studies, 0)} of studies`} />
        <Kpi k="Latency" v={`${s.median_seconds.toFixed(1)}s`} d="median per study, laptop CPU" />
      </div>
      <div className="grid g3" style={{ marginTop: 12 }}>
        <Panel title="Findings correct per study" sub="image + report">
          <HBars rows={hist.map(([k, c]) => ({ label: `${k} / 12`, c }))} series={[{ key: 'c', name: 'Studies', color: SERIES[0] }]}
                 domain={[0, Math.max(...hist.map(([, c]) => c)) + 2]} fmt={(v) => String(v)} labelWidth={70} rowH={24} />
        </Panel>
        <Panel title="Accuracy by finding" sub="live pipeline, image + report">
          <HBars rows={Object.entries(s.fused_per_label_accuracy).map(([l, v]) => ({ label: l, v })).sort((a, b) => b.v - a.v)}
                 series={[{ key: 'v', name: 'Accuracy', color: SERIES[1] }]} domain={[0.4, 1]} fmt={(v) => pct(v, 0)} labelWidth={120} rowH={22} />
        </Panel>
        <Panel title="Per-study log" sub="click a study on the Inference tab to view it">
          <div style={{ maxHeight: 320, overflowY: 'auto' }}>
            <table className="t">
              <thead><tr><th>Study</th><th className="n">Image</th><th className="n">Image + report</th><th className="n">Time</th></tr></thead>
              <tbody>
                {sorted.map((r) => (
                  <tr key={r.id}><td className="mono">{r.id}</td><td className="n">{r.image_correct}/12</td>
                    <td className="n"><b className={r.fused_correct >= 10 ? 'ok' : ''}>{r.fused_correct}/12</b></td><td className="n">{r.seconds.toFixed(1)}s</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </>
  )
}

export default function Results() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  const [view, setView] = useState('image')
  const [scatterLabel, setScatterLabel] = useState('Effusion')
  useEffect(() => { api.results().then(setData).catch((e) => setErr(e.message)) }, [])

  const m = data?.metrics, u = data?.unsupervised
  const labels = m?.labels || []
  const evalSet = useMemo(() => (m ? { image: m.final_model.test, fused: m.fusion.test, report: m.report_reader }[view] : null), [m, view])
  if (err) return <div className="panel panel-b err">{err}</div>
  if (!m) return <div className="muted">Loading results…</div>

  const final = m.final_model.test
  const comp = m.comparison
  const order = Object.keys(comp).sort((a, b) => comp[b].test_auc - comp[a].test_auc)
  const base = comp[m.baseline]
  const lift = comp.ensemble.test_auc - base.test_auc
  const ds = m.dataset

  return (
    <div>
      {/* ---------- headline ---------- */}
      <div className="grid g6">
        <Kpi k="Accuracy" v={pct(m.fusion.test.micro.accuracy)} d="image + report · all 12 findings" />
        <Kpi k="NPV" v={pct(m.fusion.test.macro.npv)} d="image + report · macro mean" />
        <Kpi k="ROC-AUC" v={f3(m.fusion.test.macro.auc)} d={`image + report · CI ${f3(m.fusion.test.macro_auc_ci95[0])}–${f3(m.fusion.test.macro_auc_ci95[1])}`} />
        <Kpi k="Image-only AUC" v={f3(final.macro.auc)} d={`accuracy ${pct(final.micro.accuracy)} · NPV ${pct(final.macro.npv)}`} />
        <Kpi k="Gain vs baseline" v={`${lift >= 0 ? '+' : ''}${f3(lift)}`} d={`image AUC over ${base.name}`} />
        <Kpi k="Test set" v={ds.test_studies} d={`radiologist-labelled · ${ds.train_studies.toLocaleString()} train`} />
      </div>
      <div className="small muted" style={{ marginTop: 6 }}>
        All test metrics are computed on held-out radiologist-labelled studies that were never used for fitting, model selection,
        hyper-parameter tuning or threshold selection. Thresholds come from 5-fold grouped cross-validation on the training pool.
      </div>

      <div style={{ marginTop: 12 }}><Strongest m={m} runs={data.inference_runs} labels={labels} /></div>
      {data.inference_runs && <InferenceRuns runs={data.inference_runs} />}

      {/* ---------- model comparison ---------- */}
      <h2 className="section">Model comparison</h2>
      <div className="grid g2">
        <Panel title="Macro ROC-AUC by model" sub="bar = held-out test · line = 95% bootstrap CI">
          <Legend items={[['Cross-validation (OOF)', SERIES[0]], ['Held-out test', SERIES[1]]]} />
          <HBars
            rows={order.map((k) => ({ label: comp[k].name, oof: comp[k].oof_auc, test: comp[k].test_auc, ci: comp[k].test_auc_ci95 }))}
            series={[{ key: 'oof', name: 'CV (OOF)', color: SERIES[0] }, { key: 'test', name: 'Test', color: SERIES[1] }]}
            domain={[0.4, 1]} ci="ci" rowH={30} />
        </Panel>
        <Panel title="Model table" sub="selected hyper-parameters">
          <div className="scroll">
            <table className="t">
              <thead><tr><th>Model</th><th className="n">CV AUC</th><th className="n">Test AUC</th><th className="n">Test F1</th><th className="n">Test Acc</th></tr></thead>
              <tbody>
                {order.map((k) => (
                  <tr key={k} className={k === 'ensemble' ? 'hl' : ''}>
                    <td title={JSON.stringify(comp[k].params)}>{comp[k].name}</td>
                    <td className="n">{f3(comp[k].oof_auc)}</td><td className="n">{f3(comp[k].test_auc)}</td>
                    <td className="n">{f3(comp[k].test_f1)}</td><td className="n">{pct(comp[k].test_accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>Ensemble = mean of Platt-calibrated probabilities from the top-3 models by CV AUC: {m.final_model.members.map((n) => comp[n].name).join(', ')}.</div>
        </Panel>
      </div>

      {/* ---------- per finding ---------- */}
      <h2 className="section">Per-finding performance</h2>
      <div className="panel">
        <div className="panel-h">
          Held-out test metrics
          <div className="seg">
            {[['image', 'Image ensemble'], ['fused', 'Image + report'], ['report', 'Report reader']].map(([k, n]) => (
              <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>{n}</button>
            ))}
          </div>
        </div>
        <div className="panel-b"><LabelTable data={evalSet} labels={labels} /></div>
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <Panel title="ROC curves" sub="held-out test">
          <Legend line items={[['Image ensemble', SERIES[0]], ['Image + report', SERIES[1]]]} />
          <div className="grid g4" style={{ gap: 10 }}>
            {labels.map((l) => (
              <MiniRoc key={l} title={l} auc={f3(final.per_label[l].auc)}
                       curves={[
                         { name: 'Image', color: SERIES[0], auc: f3(final.per_label[l].auc), ...m.roc[l] },
                         { name: 'Image + report', color: SERIES[1], auc: f3(m.fusion.test.per_label[l].auc), ...m.fusion.roc[l] },
                       ]} />
            ))}
          </div>
        </Panel>
        <Panel title="Confusion matrices" sub="image ensemble · [TN FP / FN TP]">
          <div className="grid g4" style={{ gap: 10 }}>
            {labels.map((l) => <Confusion key={l} title={l} m={final.per_label[l].confusion} />)}
          </div>
        </Panel>
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <Panel title="AUC by finding and source" sub="held-out test">
          <Legend items={[['Report reader', SERIES[2]], ['Image ensemble', SERIES[0]], ['Image + report', SERIES[1]]]} />
          <HBars rows={labels.map((l) => ({ label: l, rep: m.report_reader.per_label[l].auc, img: final.per_label[l].auc, fus: m.fusion.test.per_label[l].auc }))}
                 series={[{ key: 'rep', name: 'Report', color: SERIES[2] }, { key: 'img', name: 'Image', color: SERIES[0] }, { key: 'fus', name: 'Fused', color: SERIES[1] }]}
                 domain={[0.3, 1]} rowH={30} labelWidth={120} />
        </Panel>
        <Panel title="Label prevalence" sub="weak (train) vs radiologist (test)">
          <Legend items={[['Train · report-derived', SERIES[0]], ['Test · radiologist', SERIES[1]]]} />
          <HBars rows={labels.map((l) => ({ label: l, w: ds.weak_prevalence[l], g: ds.gold_prevalence[l] }))}
                 series={[{ key: 'w', name: 'Train', color: SERIES[0] }, { key: 'g', name: 'Test', color: SERIES[1] }]}
                 domain={[0, 0.8]} fmt={(v) => pct(v, 0)} rowH={26} labelWidth={120} />
        </Panel>
      </div>

      {/* ---------- bias / variance & tuning ---------- */}
      <h2 className="section">Bias–variance, tuning and imbalance</h2>
      <div className="grid g3">
        <Panel title="Validation curve" sub="logistic regression, CV">
          <Legend line items={[['Train AUC', SERIES[0]], ['Validation AUC', SERIES[1]]]} />
          <Lines x={m.bias_variance.map((r) => r.C)} logX xLabel="C (inverse regularisation)" fmtX={(v) => v.toExponential(0)}
                 series={[{ name: 'Train', color: SERIES[0], values: m.bias_variance.map((r) => r.train_auc) },
                          { name: 'Validation', color: SERIES[1], values: m.bias_variance.map((r) => r.val_auc) }]} />
          <div className="small muted">Left: underfitting (high bias). Right: train ↑, validation ↓ — overfitting (high variance).</div>
        </Panel>
        <Panel title="Learning curve" sub="tuned logistic regression">
          <Legend line items={[['Train AUC', SERIES[0]], ['Validation AUC', SERIES[1]]]} />
          <Lines x={m.learning_curve.map((r) => r.train_size)} xLabel="training studies"
                 series={[{ name: 'Train', color: SERIES[0], values: m.learning_curve.map((r) => r.train_auc) },
                          { name: 'Validation', color: SERIES[1], values: m.learning_curve.map((r) => r.val_auc) }]} />
          <div className="small muted">The gap narrows with more data: variance falls as the training pool grows.</div>
        </Panel>
        <Panel title="Grid search" sub="logistic regression · CV AUC">
          {(() => {
            const rows = m.tuning.logreg
            const comps = [...new Set(rows.map((r) => r.components))]
            const cs = [...new Set(rows.map((r) => r.C))]
            const vals = comps.map((k) => cs.map((c) => rows.find((r) => r.components === k && r.C === c).oof_auc))
            return <Heatmap rows={comps.map((k) => `PCA ${k}`)} cols={cs.map((c) => `C=${c}`)} values={vals} fmt={(v) => v.toFixed(3)} labelWidth={70} colLabelH={50} cell={40} title={(r, c) => `${r}, ${c}`} />
          })()}
        </Panel>
      </div>
      <div className="grid g3" style={{ marginTop: 12 }}>
        {[['knn', 'k', 'k-NN: neighbours'], ['mlp', 'alpha', 'Neural net: L2 alpha'], ['decision_tree', 'max_depth', 'Decision tree: depth']].map(([k, p, t]) => (
          <Panel key={k} title={t} sub="CV AUC">
            <Lines x={m.tuning[k].map((r) => r[p])} logX={k === 'mlp'} xLabel={p} height={160}
                   fmtX={(v) => (k === 'mlp' ? v.toExponential(0) : String(v))}
                   series={[{ name: 'CV AUC', color: SERIES[0], values: m.tuning[k].map((r) => r.oof_auc) }]} />
          </Panel>
        ))}
      </div>
      <div className="grid g2" style={{ marginTop: 12 }}>
        <Panel title="Class imbalance handling" sub="logistic regression, threshold 0.5, held-out test">
          <table className="t">
            <thead><tr><th>Weighting</th><th className="n">AUC</th><th className="n">Recall</th><th className="n">Precision</th><th className="n">Spec</th><th className="n">F1</th></tr></thead>
            <tbody>
              {Object.entries(m.imbalance).map(([k, r]) => (
                <tr key={k}><td>{k === 'balanced' ? 'Balanced class weights' : 'Unweighted'}</td><td className="n">{f3(r.test_auc)}</td><td className="n">{f3(r.recall)}</td><td className="n">{f3(r.precision)}</td><td className="n">{f3(r.specificity)}</td><td className="n">{f3(r.f1)}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 6 }}>Re-weighting leaves ranking (AUC) almost unchanged but moves the operating point toward the minority class — recall rises.</div>
        </Panel>
        <Panel title="Report reader quality" sub="rule-based labels vs radiologist, test set">
          <div className="grid g3">
            <Kpi k="Macro AUC" v={f3(m.report_reader.macro.auc)} />
            <Kpi k="Accuracy" v={pct(m.report_reader.micro.accuracy)} />
            <Kpi k="Specificity" v={pct(m.report_reader.macro.specificity)} />
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>These rules generate the training labels for {ds.train_studies} studies; their noise bounds what any image model can learn.</div>
        </Panel>
      </div>

      {/* ---------- unsupervised ---------- */}
      {u && (
        <>
          <h2 className="section">Unsupervised structure</h2>
          <div className="grid g3">
            <Panel title="PCA explained variance" sub={`${u.pca.components_for_90} PCs reach 90%`}>
              <Lines x={u.pca.cumulative.map((_, i) => i + 1).filter((_, i) => i % 4 === 0)} xLabel="components" markers={false} yDomain={[0, 1]}
                     fmtY={(v) => pct(v, 0)} series={[{ name: 'Cumulative', color: SERIES[0], values: u.pca.cumulative.filter((_, i) => i % 4 === 0) }]} />
            </Panel>
            <Panel title="K-means: elbow" sub="inertia vs k">
              <Lines x={u.kmeans.sweep.map((r) => r.k)} xLabel="k" fmtY={(v) => `${(v / 1e6).toFixed(2)}M`}
                     series={[{ name: 'Inertia', color: SERIES[0], values: u.kmeans.sweep.map((r) => r.inertia) }]} />
            </Panel>
            <Panel title="K-means: silhouette" sub={`best k = ${u.kmeans.best_k}`}>
              <Lines x={u.kmeans.sweep.map((r) => r.k)} xLabel="k"
                     series={[{ name: 'Silhouette', color: SERIES[1], values: u.kmeans.sweep.map((r) => r.silhouette) }]} />
            </Panel>
          </div>
          <div className="grid g2" style={{ marginTop: 12 }}>
            <Panel title="SVD projection" sub="1,200 training studies">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Legend items={[['Negative', SERIES[0]], [`${scatterLabel} positive`, SERIES[1]]]} />
                <select value={scatterLabel} onChange={(e) => setScatterLabel(e.target.value)} className="btn" style={{ padding: '2px 6px' }}>
                  {labels.map((l) => <option key={l}>{l}</option>)}
                </select>
              </div>
              <Scatter x={u.svd_2d.x} y={u.svd_2d.y} cls={u.svd_2d.labels[scatterLabel]} names={['Negative', `${scatterLabel} positive`]} />
            </Panel>
            <Panel title="Clustering summary" sub="50 PCA components">
              <table className="t">
                <thead><tr><th>Method</th><th>Setting</th><th className="n">Silhouette</th><th className="n">ARI vs K-means</th></tr></thead>
                <tbody>
                  <tr><td>K-means</td><td>k = {u.kmeans.best_k}</td><td className="n">{f3(Math.max(...u.kmeans.sweep.map((r) => r.silhouette)))}</td><td className="n">1.000</td></tr>
                  <tr><td>K-medoids</td><td>k = {u.kmedoids.k}</td><td className="n">{f3(u.kmedoids.silhouette)}</td><td className="n">{f3(u.kmedoids.agreement_with_kmeans_ari)}</td></tr>
                  {Object.entries(u.hierarchical).map(([k, r]) => (
                    <tr key={k}><td>Hierarchical</td><td>{k} linkage</td><td className="n">{f3(r.silhouette)}</td><td className="n">{f3(r.ari_vs_kmeans)}</td></tr>
                  ))}
                  <tr><td>DBSCAN</td><td>eps {u.dbscan.eps.toFixed(1)}, minPts {u.dbscan.min_samples}</td><td className="n" colSpan={2}>{u.dbscan.clusters} clusters · {pct(u.dbscan.noise_fraction, 0)} noise</td></tr>
                </tbody>
              </table>
              <div className="small muted" style={{ marginTop: 8 }}>
                Agreement of 2-means with clinical groups (ARI): osteoarthritis {f3(u.cluster_vs_clinical.ari_kmeans2_vs_oa)},
                acute trauma {f3(u.cluster_vs_clinical.ari_kmeans2_vs_trauma)}. Low values show the embedding's dominant structure is acquisition, not pathology —
                why supervised heads are needed.
              </div>
            </Panel>
          </div>
          <div className="grid g2" style={{ marginTop: 12 }}>
            <Panel title="K-means cluster profiles" sub="finding prevalence per cluster">
              <Heatmap rows={u.kmeans.profiles.map((p) => `C${p.cluster} (n=${p.size})`)} cols={labels}
                       values={u.kmeans.profiles.map((p) => labels.map((l) => p.prevalence[l]))} domain={[0, 0.8]}
                       fmt={(v) => v.toFixed(2)} labelWidth={90} cell={36} />
            </Panel>
            <Panel title="Finding co-occurrence" sub="P(A and B), training labels">
              <Heatmap rows={labels} cols={labels} values={u.cooccurrence} fmt={(v) => v.toFixed(2)} cell={30} labelWidth={110} />
            </Panel>
          </div>

          <h2 className="section">Association rules (Apriori)</h2>
          <Panel title={`Top rules by lift`} sub={`min support ${u.apriori.min_support}, min confidence ${u.apriori.min_confidence} · ${u.apriori.rule_count} rules over ${u.apriori.transactions} studies`}>
            <div className="scroll">
              <table className="t">
                <thead><tr><th>If</th><th>Then</th><th className="n">Support</th><th className="n">Confidence</th><th className="n">Lift</th></tr></thead>
                <tbody>
                  {u.apriori.rules.slice(0, 14).map((r, i) => (
                    <tr key={i}><td>{r.antecedent.join(' + ')}</td><td>{r.consequent.join(' + ')}</td>
                      <td className="n">{f3(r.support)}</td><td className="n">{f3(r.confidence)}</td><td className="n">{r.lift.toFixed(2)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </div>
  )
}
