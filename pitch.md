# KneeML — pitch

## The problem
A knee MRI has hundreds of images across three planes, and a radiologist reads every one to
report a dozen possible injuries: torn ligaments and menisci, osteoarthritis, fluid, cysts,
bone bruises, fractures. Expert-labelled data is scarce: in our collection only
**58 of 4,407 studies** carry radiologist labels.

## Our idea
Learn from what hospitals already have: **the free-text report**.

1. A transparent, multilingual **rule-based report reader** turns 4,349 reports
   in 10+ languages into training labels, with no manual annotation and no external AI service.
2. A frozen, pretrained vision model turns each MRI into a compact numeric fingerprint (24 slices → 4,611 numbers).
3. Classical, explainable ML (logistic regression, SVM, Random Forest, boosting, k-NN, an MLP)
   learns the 12 findings; the best three are combined into a calibrated ensemble.

## Results (held-out, radiologist-labelled studies, never seen in training)
* **80.6% accuracy** across all 12 findings and **88.5% negative predictive value** (image + report).
* Macro ROC-AUC **0.845** with image + report, vs 0.808 for the report rules alone and 0.707 for images alone: the two sources are complementary.
* Images alone: 66.7% accuracy, 81.1% NPV.
* Strongest findings (image + report AUC): MCL 0.948, Medial Meniscus 0.918, ACL 0.913, Lateral Meniscus 0.894.
* **+0.002 AUC** from the ensemble over a logistic-regression baseline.
* The best single finding from images alone is Medial OA (AUC 0.777).

* Run end-to-end through the web app on all 58 held-out studies: **80.3%** of 696 decisions correct, 31 studies with ≥ 10/12 findings right, 5 perfect, median 1.9 s per study on a laptop CPU.

## Why it matters
* **Cheap**: one ≈35-minute GPU job; everything else trains in minutes on a laptop CPU.
* **Explainable**: every stage is a classical, inspectable model; rules are readable; thresholds are per finding.
* **Honest evaluation**: expert-labelled studies are held out completely; confidence intervals reported.
* **Usable**: a web app shows the MRI, per-finding probabilities and decision thresholds, and
  accepts new studies (PNG, JPG or DICOM) in about 3 seconds on a CPU.

## Beyond the classifier
* Unsupervised analysis (PCA, SVD, K-means, K-medoids, hierarchical, DBSCAN) shows the image
  embeddings cluster by scanner protocol, not disease, which is why supervision matters.
* Apriori mining over 4,407 studies recovers clinical patterns, e.g.
  *Fracture → Contusion* (lift 2.86).

## What's next
Fine-tune the encoder on MRI, learn a text model on top of the rules, add 3D context, and
grow the expert-labelled test set to tighten the confidence intervals.

*Educational project, not a medical device.*
