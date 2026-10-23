const STEPS = [
  ['01', 'Series selection', 'Per study pick the most fluid-sensitive, fat-suppressed sagittal, coronal and axial series.'],
  ['02', 'Slice sampling', 'Order slices by patient geometry; take 8 evenly spaced from the central 80%. Left knees mirrored to right.'],
  ['03', 'Normalisation', 'Clip to 1st–99th percentile per series, letterbox to 256×256 uint8.'],
  ['04', 'Frozen encoder', 'DINOv2 ViT-S/14 (no fine-tuning) → CLS + mean-patch token = 768-d per slice.'],
  ['05', 'Study vector', 'Mean and max pooling over slices per plane + plane flags → 4,611-d.'],
  ['06', 'Classical heads', 'Standardise → PCA → 9 classifiers, one-vs-rest over 12 findings; tuned by grouped 5-fold CV.'],
  ['07', 'Ensemble', 'Platt-calibrate each model, average the top-3; per-finding thresholds maximise CV F1.'],
]

const SYLLABUS = [
  ['I', 'Supervised / unsupervised learning, inductive bias', 'Weak supervision from a hand-written report reader (an explicit inductive bias); clustering on unlabeled embeddings.'],
  ['I', 'Bias, variance, under/overfitting', 'Validation curve over regularisation C; learning curve over training-set size.'],
  ['I', 'Logistic regression', 'Baseline and ensemble member; Platt scaling for calibration.'],
  ['II', 'Decision trees, Random Forest', 'Depth sweep for a single tree; bagged forest with balanced bootstrap weights.'],
  ['II', 'Train/test split, cross-validation', 'Radiologist-labelled studies held out; GroupKFold(5) grouped by report text.'],
  ['II', 'Confusion matrix, accuracy, precision, recall, F1', 'Reported per finding plus specificity, NPV and macro / micro means.'],
  ['III', 'Bayes theorem, Naive Bayes, MLE', 'Gaussian NB (class-conditional MLE); logistic models fitted by maximum likelihood.'],
  ['III', 'Ensembles: bagging, boosting', 'Random Forest (bagging), histogram gradient boosting, soft-voting ensemble.'],
  ['III', 'SVM: linear, non-linear kernels', 'Linear SVM and RBF-kernel SVM with tuned C.'],
  ['III', 'k-NN', 'Cosine-distance, distance-weighted k-NN with k tuned.'],
  ['III', 'Hyper-parameter tuning, imbalanced data', 'Grid search on CV AUC; class-weight balancing; per-finding thresholds.'],
  ['IV', 'K-means, K-medoids, hierarchical, density-based', 'Elbow + silhouette for k; PAM-style K-medoids; Ward/average/complete linkage; DBSCAN with k-distance eps.'],
  ['V', 'Association rules, Apriori, support, confidence', 'Apriori over co-occurring findings; rules ranked by lift.'],
  ['V', 'Dimensionality reduction: PCA, SVD', 'PCA inside every model pipeline; explained-variance curve; truncated SVD projection.'],
  ['VI', 'ANN, feed-forward networks', 'MLP classifier (one hidden layer, L2-regularised, early stopping).'],
]

export default function Method() {
  return (
    <div>
      <h2 className="section" style={{ marginTop: 4 }}>Pipeline</h2>
      <div className="flow">
        {STEPS.map(([i, t, d]) => (
          <div key={i} className="st"><span className="idx">{i}</span><b>{t}</b><span>{d}</span></div>
        ))}
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <div className="panel">
          <div className="panel-h">Data and labels</div>
          <div className="panel-b small" style={{ display: 'grid', gap: 6 }}>
            <div>4,407 knee MRI studies (≈24k series) with free-text reports in 10+ languages; 58 studies carry radiologist labels for 12 findings.</div>
            <div><b>Training labels</b> come from a deterministic multilingual rule-based report reader (anatomy × lesion patterns, clause-level negation and hedging).</div>
            <div><b>Test labels</b> are the radiologist annotations, held out completely.</div>
            <div><b>Image + report fusion</b> averages the image probability with the report reader score (fixed 50/50, chosen a priori).</div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-h">Engineering</div>
          <div className="panel-b small" style={{ display: 'grid', gap: 6 }}>
            <div>Feature extraction runs once as a Kaggle GPU kernel (≈1 h for all studies); everything else runs on a laptop CPU.</div>
            <div>Backend: FastAPI serving the scikit-learn ensemble and a CPU DINOv2 encoder (~3 s per study).</div>
            <div>Frontend: React + Vite, hand-built SVG charts, no UI framework.</div>
            <div>Full write-up: <span className="mono">technical.md</span> in the repository.</div>
          </div>
        </div>
      </div>

      <h2 className="section">Syllabus coverage</h2>
      <div className="panel">
        <div className="scroll">
          <table className="t">
            <thead><tr><th>Unit</th><th>Topic</th><th>Where it is used</th></tr></thead>
            <tbody>
              {SYLLABUS.map(([u, t, w]) => (
                <tr key={t}><td className="mono">{u}</td><td>{t}</td><td style={{ whiteSpace: 'normal' }}>{w}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
