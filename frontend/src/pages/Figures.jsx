import { useEffect, useState } from 'react'

// Every figure in results/figures with a plain-language explanation.
const FIGURES = [
  {
    file: '01_model_comparison.png', group: 'Performance', title: 'Model comparison',
    what: 'Macro ROC-AUC (average over the 12 findings) for every classifier that was tried, on cross-validation and on the held-out test set.',
    read: 'Longer bar = better at ranking abnormal studies above normal ones. 0.5 is a coin flip, 1.0 is perfect. Similar CV and test bars mean the model generalises.',
    why: 'Justifies the final choice: the calibrated ensemble of the top models beats each single model and the logistic-regression baseline.',
  },
  {
    file: '02_per_label_metrics.png', group: 'Performance', title: 'Metrics per finding',
    what: 'A heatmap of AUC, accuracy, precision, recall, specificity and F1 for each of the 12 findings on the held-out test studies (image ensemble).',
    read: 'Each row is a finding, each column a metric; darker blue = higher score. Recall = how many real cases were caught; precision = how many flagged cases were real.',
    why: 'Shows the model is not equally good at everything: large, high-contrast findings (e.g. effusion) are easier than subtle ones.',
  },
  {
    file: '03_roc_curves.png', group: 'Performance', title: 'ROC curves',
    what: 'One ROC curve per finding. The title shows AUC for image only / image + report.',
    read: 'A curve hugging the top-left corner is good; the diagonal is random guessing. Each point is a different decision threshold.',
    why: 'Shows the trade-off between catching cases (true positives) and false alarms, and how much adding the report helps.',
  },
  {
    file: '04_confusion_matrices.png', group: 'Performance', title: 'Confusion matrices',
    what: 'For each finding: counts of true negatives, false positives, false negatives and true positives on the test set.',
    read: 'The diagonal (top-left, bottom-right) is correct. Off-diagonal cells are mistakes; bottom-left = missed cases.',
    why: 'Makes errors concrete: you can see exactly how many studies were missed or over-called per finding.',
  },
  {
    file: '05_bias_variance.png', group: 'Model behaviour', title: 'Bias vs variance',
    what: 'Left: validation curve over regularisation strength C. Right: learning curve as the training set grows.',
    read: 'A gap between train and validation lines = overfitting (variance). Both lines low = underfitting (bias). Validation still rising at the right = more data would help.',
    why: 'Explains how the regularisation level was picked and whether collecting more data is worthwhile.',
  },
  {
    file: '06_sources.png', group: 'Model behaviour', title: 'Image vs report vs fused',
    what: 'Test AUC for each finding using the images alone, the report text alone, and both combined.',
    read: 'Compare the three bars per finding. Where the fused bar is highest, the two sources complement each other.',
    why: 'Shows why the app fuses image and report: each source covers the other\'s blind spots.',
  },
  {
    file: '07_pca_kmeans.png', group: 'Unsupervised', title: 'PCA and K-means',
    what: 'Cumulative variance explained by principal components, and the elbow / silhouette scores for K-means clustering.',
    read: 'PCA: where the curve flattens, extra components add little. Elbow: the bend suggests a good k. Silhouette: higher = cleaner clusters.',
    why: 'Shows how far the 4,611-d study vectors can be compressed and whether the studies form natural groups.',
  },
  {
    file: '08_svd.png', group: 'Unsupervised', title: 'SVD projection',
    what: 'Every study embedding squeezed to 2 dimensions with truncated SVD.',
    read: 'Each dot is a study. Dots close together have similar image features; colour shows whether the knee has medial osteoarthritis.',
    why: 'A quick visual check that the frozen DINOv2 features carry structure related to the findings.',
  },
  {
    file: '09_cooccurrence.png', group: 'Data', title: 'Finding co-occurrence',
    what: 'Probability that two findings appear in the same knee, P(A and B).',
    read: 'Darker cell = the pair appears together more often. The diagonal is how common each finding is on its own.',
    why: 'Explains correlated predictions (e.g. ACL tears with bone contusions) and feeds the association-rule analysis.',
  },
  {
    file: '10_grid_search.png', group: 'Model behaviour', title: 'Grid search',
    what: 'Cross-validated AUC of logistic regression for each combination of hyper-parameters tried.',
    read: 'Each cell is one setting with its CV AUC written inside; the darkest cell (highest AUC) is the selected combination.',
    why: 'Shows the hyper-parameters were chosen systematically on CV, never on the test set.',
  },
]
const GROUPS = ['All', ...new Set(FIGURES.map((f) => f.group))]
const src = (f) => `/api/figures/${f}`

export default function Figures() {
  const [group, setGroup] = useState('All')
  const [open, setOpen] = useState(null)
  const shown = FIGURES.filter((f) => group === 'All' || f.group === group)

  useEffect(() => {
    if (open == null) return
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(null)
      if (e.key === 'ArrowRight') setOpen((i) => (i + 1) % FIGURES.length)
      if (e.key === 'ArrowLeft') setOpen((i) => (i - 1 + FIGURES.length) % FIGURES.length)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  const cur = open != null ? FIGURES[open] : null
  return (
    <div>
      <div className="fig-head">
        <div>
          <h2 className="section" style={{ margin: 0 }}>Figures</h2>
          <div className="small muted">All charts produced by the training pipeline (<span className="mono">results/figures</span>), each with a short explanation. Click a figure to enlarge.</div>
        </div>
        <div className="seg">
          {GROUPS.map((g) => <button key={g} className={group === g ? 'on' : ''} onClick={() => setGroup(g)}>{g}</button>)}
        </div>
      </div>

      <div className="figs">
        {shown.map((f) => (
          <div key={f.file} className="panel fig">
            <div className="panel-h">{f.title} <span className="sub">{f.group}</span></div>
            <button className="fig-img" onClick={() => setOpen(FIGURES.indexOf(f))} aria-label={`Enlarge ${f.title}`}>
              <img src={src(f.file)} alt={f.title} loading="lazy" />
            </button>
            <div className="panel-b fig-notes">
              <p><b>What it shows.</b> {f.what}</p>
              <p><b>How to read it.</b> {f.read}</p>
              <p className="why"><b>Why it matters.</b> {f.why}</p>
            </div>
          </div>
        ))}
      </div>

      {cur && (
        <div className="lightbox" onClick={() => setOpen(null)}>
          <div className="lb-card" onClick={(e) => e.stopPropagation()}>
            <div className="panel-h">
              {cur.title}
              <span style={{ display: 'flex', gap: 6 }}>
                <button className="iconbtn" onClick={() => setOpen((open - 1 + FIGURES.length) % FIGURES.length)}>← Prev</button>
                <button className="iconbtn" onClick={() => setOpen((open + 1) % FIGURES.length)}>Next →</button>
                <button className="iconbtn" onClick={() => setOpen(null)}>Close</button>
              </span>
            </div>
            <div className="lb-body">
              <img src={src(cur.file)} alt={cur.title} />
              <div className="fig-notes small">
                <p><b>What it shows.</b> {cur.what}</p>
                <p><b>How to read it.</b> {cur.read}</p>
                <p className="why"><b>Why it matters.</b> {cur.why}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
