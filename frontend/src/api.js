async function req(path, opts) {
  const res = await fetch(path, opts)
  if (!res.ok) {
    let msg = res.statusText
    try { msg = (await res.json()).detail || msg } catch { /* not json */ }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  health: () => req('/api/health'),
  labels: () => req('/api/labels'),
  samples: () => req('/api/samples'),
  results: () => req('/api/results'),
  predictSample: (id, useReport) => req(`/api/predict/sample/${id}?use_report=${useReport}`, { method: 'POST' }),
  predictUpload: (form) => req('/api/predict/upload', { method: 'POST', body: form }),
  sliceUrl: (id, plane, k) => `/api/samples/${id}/slice/${plane.toLowerCase()}/${k}`,
}

export const pct = (v, d = 1) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`)
export const f3 = (v) => (v == null ? '–' : v.toFixed(3))
