import { useEffect, useState } from 'react'
import { api } from './api.js'
import { TipProvider } from './charts.jsx'
import Inference from './pages/Inference.jsx'
import Results from './pages/Results.jsx'
import Method from './pages/Method.jsx'

const TABS = [['inference', 'Inference'], ['results', 'Results'], ['method', 'Method']]

function initialTab() {
  const h = window.location.hash.replace('#', '')
  return TABS.some(([k]) => k === h) ? h : 'inference'
}

export default function App() {
  const [tab, setTab] = useState(initialTab)
  const [health, setHealth] = useState(null)
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('theme') || '' } catch { return '' }
  })

  useEffect(() => { api.health().then(setHealth).catch(() => setHealth({ ok: false })) }, [])
  useEffect(() => { window.location.hash = tab }, [tab])
  useEffect(() => {
    const onHash = () => setTab(initialTab())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
    try { localStorage.setItem('theme', theme) } catch { /* storage unavailable */ }
  }, [theme])

  const dark = theme ? theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches

  return (
    <TipProvider>
      <nav className="nav">
        <div className="brand">KneeML <small>Knee MRI abnormality detection</small></div>
        <div className="tabs">
          {TABS.map(([k, label]) => (
            <button key={k} className={`tab ${tab === k ? 'on' : ''}`} onClick={() => setTab(k)}>{label}</button>
          ))}
        </div>
        <div className="status">
          <span className={`dot ${health?.ok && health?.model ? 'ok' : ''}`} />
          {health == null ? 'connecting…' : health.ok ? (health.model ? 'model loaded' : 'model not trained') : 'backend offline'}
          <button className="iconbtn" onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label="Toggle theme">
            {dark ? 'Light' : 'Dark'}
          </button>
        </div>
      </nav>
      <main>
        {tab === 'inference' && <Inference />}
        {tab === 'results' && <Results />}
        {tab === 'method' && <Method />}
      </main>
    </TipProvider>
  )
}
