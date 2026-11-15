\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% Draft. Compile with pdflatex; figures are read from results/figures/ (set \graphicspath below).

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{array}
\usepackage{booktabs}
\graphicspath{{results/figures/}}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{KneeML: Multi-Label Knee MRI Abnormality Detection from Report-Derived Weak Labels using Frozen Vision Features and Calibrated Classical Classifiers}

\author{
\IEEEauthorblockN{1\textsuperscript{st} Prof. Swapnil K. Shinde}
\IEEEauthorblockA{\textit{Dept. of AI\&DS} \\
\textit{Vishwakarma Institute of Technology}\\
Pune, India \\
swapnil.shinde@vit.edu}
\and
\IEEEauthorblockN{2\textsuperscript{nd} Siddhesh K. Chaudhari}
\IEEEauthorblockA{\textit{Dept. of AI\&DS} \\
\textit{Vishwakarma Institute of Technology}\\
Pune, India \\
siddhesh.chaudhari241@vit.edu}
\and
\IEEEauthorblockN{3\textsuperscript{rd} Shishir R. Chandanshiv}
\IEEEauthorblockA{\textit{Dept. of AI\&DS} \\
\textit{Vishwakarma Institute of Technology}\\
Pune, India \\
shishir.chandanshiv24@vit.edu}
\and
\IEEEauthorblockN{4\textsuperscript{th} Om K. Chavan}
\IEEEauthorblockA{\textit{Dept. of AI\&DS} \\
\textit{Vishwakarma Institute of Technology}\\
Pune, India \\
om.chavan24@vit.edu}
\and
\IEEEauthorblockN{5\textsuperscript{th} Arya S. Deshmukh}
\IEEEauthorblockA{\textit{Dept. of AI\&DS} \\
\textit{Vishwakarma Institute of Technology}\\
Pune, India \\
arya.deshmukh241@vit.edu}
\and
\IEEEauthorblockN{6\textsuperscript{th} Atharv R. Dhole}
\IEEEauthorblockA{\textit{Dept. of AI\&DS} \\
\textit{Vishwakarma Institute of Technology}\\
Pune, India \\
atharv.dhole241@vit.edu}
}

\maketitle

\begin{abstract}
We present KneeML, a scalable, report-supervised pipeline for automated reading of knee magnetic resonance imaging (MRI), built on data derived from a complete ~250~GB multi-centre archive of 4,407 studies. KneeML is a pipeline that detects 12 knee abnormalities (cruciate and collateral ligament tears, medial and lateral meniscal tears, compartmental osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion and fracture). A transparent, multilingual rule-based reader converts free-text reports written in more than ten languages into training labels for 4,349 studies without manual annotation. Twenty-four fluid-sensitive slices per study are embedded by a frozen DINOv2 vision transformer and pooled into a 4,611-dimensional study vector. Nine classical classifiers are compared under grouped cross-validation, and the three best are combined into a Platt-calibrated soft-voting ensemble with per-finding thresholds. On an independent radiologist-verified test set, image and report fusion achieves a macro ROC-AUC of 0.845 (95\% CI 0.805--0.881), 80.6\% accuracy over 696 decisions and 88.5\% negative predictive value; ligament and meniscus findings reach AUCs of 0.89--0.95, and negative calls for ACL, MCL and Baker's cyst are correct 97--98\% of the time. The full system requires a single 35-minute GPU job and otherwise trains and runs on a laptop CPU in about two seconds per study.
\end{abstract}

\begin{IEEEkeywords}
knee MRI, weak supervision, radiology report mining, multi-label classification, self-supervised features, DINOv2, support vector machine, model calibration, explainable machine learning.
\end{IEEEkeywords}

\section{Introduction}

MRI is the reference non-invasive examination for internal derangement of the knee because it depicts ligaments, menisci, cartilage, synovium and bone marrow that radiographs cannot show. A single knee examination contains several image series in the sagittal, coronal and axial planes, and the radiologist must assess a dozen structures for each study. Computer-aided reading could triage urgent studies, act as a second reader, and support settings where musculoskeletal radiologists are scarce.

Deep learning has shown strong results on knee MRI when large expert-labelled datasets are available. The MRNet study \cite{b1} trained convolutional networks on 1,370 examinations to detect anterior cruciate ligament (ACL) tears, meniscal tears and general abnormality, and subsequent work refined the architecture \cite{b2}\cite{b3}. These systems, however, target at most three findings and depend on thousands of expert labels. In most real collections the situation is the reverse: images and free-text reports are plentiful, and expert image-level labels are rare.

The collection used in this work is a large, realistic example of this setting: approximately 250~GB of DICOM covering 4,407 knee MRI studies from multiple centres and scanners, each accompanied by a free-text radiology report in one of more than ten languages. KneeML turns this entire archive into training data without manual annotation.

This paper therefore asks whether a practical 12-finding knee MRI classifier can be built by (i) deriving training labels automatically from the existing reports, (ii) describing the images with a frozen, general-purpose pretrained encoder, and (iii) learning with classical, inspectable classifiers that train in minutes. The main contributions are:
\begin{enumerate}
    \item A transparent, multilingual rule-based report reader with negation and uncertainty handling that produces 12 report-derived labels for 4,349 studies and agrees with radiologists at 80.3\% accuracy and 0.808 macro AUC.
    \item A reproducible image representation: per-plane series selection, slice sampling, intensity and laterality normalisation, and frozen DINOv2 embeddings pooled into a 4,611-dimensional study vector.
    \item A systematic comparison of nine classical classifiers under grouped cross-validation, with Platt calibration, per-finding thresholds and a soft-voting ensemble.
    \item Late fusion of image and report evidence, improving held-out macro AUC from 0.707 (images) and 0.808 (reports) to 0.845.
    \item An end-to-end web application that reproduces the evaluation at 80.3\% accuracy with a median latency of 1.9~s per study on a laptop CPU and explains each prediction in plain language.
\end{enumerate}

\section{Literature Review}

\textit{Deep learning for knee MRI.} Bien et al. \cite{b1} introduced MRNet, in which an ImageNet-pretrained CNN extracts features from every slice of each plane, the features are pooled per plane, and logistic regression combines the planes; reported AUCs were 0.965 for ACL tears, 0.847 for meniscal tears and 0.937 for abnormality. ELNet \cite{b2} reached comparable performance with a lighter network, and Azcona et al. \cite{b3} compared several architectures on the same data. Liu et al. \cite{b4} detected cartilage lesions, and Pedoia et al. \cite{b5} used 3D CNNs to stage meniscal and patellofemoral degeneration. All of these depend on large expert-annotated cohorts and address a small number of findings.

\textit{Labels from radiology reports.} Large chest radiograph datasets were constructed by mining reports rather than annotating images. ChestX-ray8 \cite{b6} and CheXpert \cite{b7} used rule-based natural language processing that detects mentions and then classifies them as negated or uncertain, building on the NegEx algorithm \cite{b8}. CheXbert \cite{b9} later trained a transformer on rule-generated and expert labels. Snorkel \cite{b10} formalised training on programmatically generated, noisy labels. These tools are English-only; the reports in our collection are written in at least ten languages.

\textit{Pretrained features for medical imaging.} Raghu et al. \cite{b11} showed that transfer from natural images is helpful in medical imaging mainly through feature reuse, which motivates freezing a pretrained encoder when labels are scarce. Self-supervised vision transformers such as DINOv2 \cite{b12} produce general-purpose features on which simple linear classifiers perform well.

\textit{Classical learning and calibration.} Support vector machines \cite{b13}, random forests \cite{b14} and logistic regression remain strong on fixed-length features with limited labels. Platt scaling \cite{b15} converts classifier scores to probabilities, which is required for meaningful averaging in soft-voting ensembles.

\textit{Gap.} No prior work above combines multilingual report-derived labels, a frozen self-supervised encoder and calibrated classical classifiers for a 12-finding knee MRI task, evaluated against radiologist labels that were never used for development. This work fills that gap with a system that is cheap to train and transparent at every stage.

\section{Dataset}

\subsection{Source Collection}
The source is a public, de-identified, multi-centre knee MRI collection with free-text reports (approximately 250~GB of DICOM). After loading, it contains 4,407 studies and 24,371 series: 9,864 sagittal, 8,609 coronal and 5,898 axial. Reports are written in Spanish, English, German, Dutch, French, Portuguese, Turkish, Croatian, Bulgarian, Greek and other languages. The raw collection serves as the reference from which the working dataset is derived once.

\subsection{Working Dataset and Splits}
The working dataset is derived from, and follows, the complete source archive. A radiologist-verified subset of 58 studies forms an independent \textbf{test set} that is never used for fitting, model selection, hyper-parameter tuning, threshold selection or ensemble construction. The remaining 4,349 studies form the \textbf{training pool}, labelled automatically from their reports (Section~\ref{sec:labels}). Studies that share an identical report text (for example, bilateral examinations) are assigned to the same group (4,273 groups) so that grouped cross-validation does not leak information between folds.

Table~\ref{tab:prev} lists the 12 findings with their prevalence. Prevalence spans an order of magnitude (MCL 5.6\%, effusion 57.5\% in training), which KneeML addresses with class weighting and per-finding thresholds. The test set is particularly demanding, with 4.1 findings per knee on average.

\begin{table}[!t]
\caption{Target Findings and Prevalence}
\label{tab:prev}
\begin{center}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Finding} & \textbf{Train (weak)} & \textbf{Test (expert)} \\
\hline
ACL tear & 13.0\% & 41.4\% \\
MCL tear & 5.6\% & 15.5\% \\
Medial meniscus tear & 31.1\% & 44.8\% \\
Lateral meniscus tear & 13.2\% & 39.7\% \\
Medial OA & 30.1\% & 25.9\% \\
Lateral OA & 26.4\% & 19.0\% \\
Patellofemoral OA & 53.2\% & 36.2\% \\
Effusion & 57.5\% & 60.3\% \\
Synovitis & 11.8\% & 46.6\% \\
Baker's cyst & 27.6\% & 20.7\% \\
Bone contusion & 24.9\% & 32.8\% \\
Fracture & 6.6\% & 31.0\% \\
\hline
\textbf{Studies} & \textbf{4,349} & \textbf{58} \\
\hline
\end{tabular}
\end{center}
\end{table}

\section{Methodology}

The pipeline (Fig.~\ref{fig:pipeline}) has two branches that meet at the classifier: a text branch that turns reports into report-derived labels, and an image branch that turns each study into a fixed-length vector.

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{00_pipeline.png}}
\caption{KneeML pipeline. Reports are converted to report-derived labels; 24 slices per study are embedded by a frozen encoder and pooled; classical classifiers are trained on the resulting vectors and optionally fused with the report score.}
\label{fig:pipeline}
\end{figure}

\subsection{Weak Labels from Reports}
\label{sec:labels}
The report reader is a deterministic rule system written as an explicit, human-readable hypothesis. Text is lower-cased, accents are stripped and language-specific characters normalised. The report is split into clauses, and clauses that are questions or clinical history (for example ``Indication:'', ``Antecedentes'', ``Fragestellung'') are discarded. A finding is mentioned when an anatomy pattern and a lesion pattern co-occur in a clause (e.g., \textit{medial meniscus} with \textit{tear / rotura / Riss / scheur / y\i rt\i k}), or when a stand-alone pattern matches (effusion, Baker's cyst, contusion, fracture, synovitis). Each mention $m$ receives a score
\begin{equation}
s(m) = \begin{cases} 1.0 & \text{asserted} \\ 0.5 & \text{hedged (suspected, possible, grade 1)} \\ 0.0 & \text{negated (no, without, sin, kein, intact)} \end{cases}
\end{equation}
The study score for finding $k$ is $r_k = \max_m s(m)$ (zero if never mentioned), and the training label is $y_k = \mathbb{1}[r_k \ge 0.75]$.

\subsection{Image Preprocessing}
For each plane, the series maximising $2\cdot\text{fluid-sensitive} + \text{fat-suppressed}$ is selected, because such sequences depict tears, oedema and fluid with high contrast. Slices are ordered by projecting the image position onto the slice normal, and eight slices are sampled uniformly from the central 80\% of the stack. Intensities are clipped to the 1st--99th percentile of the series, scaled to $[0,1]$, and letterboxed to $256\times256$. Left knees are mirrored so that medial and lateral structures appear on the same side for every study.

\subsection{Frozen Feature Extraction}
Each slice is resized to 224 pixels and passed through a frozen DINOv2 ViT-S/14 \cite{b12}. The class token and the mean of the patch tokens are concatenated into a 768-dimensional slice descriptor. For plane $p$ with slice descriptors $\mathbf{f}_{p,1},\dots,\mathbf{f}_{p,8}$, the study vector is
\begin{equation}
\mathbf{x} = \big[\,\text{mean}_i\,\mathbf{f}_{p,i},\ \max_i\,\mathbf{f}_{p,i}\,\big]_{p\in\{\text{sag},\text{cor},\text{ax}\}} \oplus \mathbf{e},
\end{equation}
where $\mathbf{e}\in\{0,1\}^3$ flags which planes are present, giving $d=4{,}611$. Mean pooling summarises the plane; max pooling preserves a local signal that appears on few slices. Extraction ran once as two GPU jobs of about 17 minutes each with no failures. A CPU re-implementation used by the web application reproduces the GPU embeddings (mean cosine similarity 0.987).

\subsection{Classifiers}
Every model is the pipeline StandardScaler $\rightarrow$ PCA $\rightarrow$ estimator, applied one-vs-rest to the 12 findings. Nine estimators were compared (Table~\ref{tab:models}): logistic regression (baseline), Gaussian naive Bayes, cosine $k$-nearest neighbours, linear SVM, RBF-kernel SVM, a decision tree, a random forest with balanced bootstrap weights, histogram gradient boosting, and a one-hidden-layer multilayer perceptron with early stopping. Hyper-parameters were selected by grid search on out-of-fold macro AUC.

\subsection{Validation, Calibration and Thresholds}
Model selection uses five-fold GroupKFold on the training pool. Each model's out-of-fold scores are mapped to probabilities by per-finding Platt scaling \cite{b15}. For each finding $k$ the decision threshold $\tau_k$ maximises out-of-fold F1. Class-balanced weights compensate for prevalence differences.

\subsection{Ensemble and Fusion}
The three models with the highest cross-validated AUC (RBF SVM, linear SVM and logistic regression) form a soft-voting ensemble $\hat{p}_k = \tfrac{1}{3}\sum_{j} \hat{p}^{(j)}_k$. When a report is available, image and report evidence are fused with a weight fixed in advance:
\begin{equation}
\tilde{p}_k = (1-w)\,\hat{p}_k + w\, r_k, \qquad w = 0.5,
\end{equation}
with threshold $\tilde{\tau}_k = (1-w)\tau_k + 0.75\,w$. No parameter of the fusion was tuned on the test set.

\subsection{Unsupervised and Association Analysis}
To characterise the representation, PCA, truncated SVD, $k$-means, $k$-medoids, hierarchical clustering and DBSCAN \cite{b18} were applied to the study vectors, and Apriori association rules \cite{b17} were mined over the findings of all studies (minimum support 0.05, minimum confidence 0.5).

\subsection{Evaluation}
Metrics are computed on the 58 held-out studies (696 binary decisions): macro and micro ROC-AUC, accuracy, precision, recall, specificity, negative predictive value (NPV) and F1. Confidence intervals for macro AUC use 1,000 bootstrap resamples of studies.

\section{Results and Performance Evaluation}

\subsection{Experimental Setup}
Feature extraction ran on a single NVIDIA T4 GPU; all training, analysis and inference ran on a laptop CPU with scikit-learn \cite{b16}. The full supervised experiment suite completes in about 12 minutes.

\subsection{Model Comparison}
Table~\ref{tab:models} compares the classifiers on image features alone. Margin-based linear and kernel models make the best use of the rich 4,611-dimensional representation. The calibrated ensemble achieves the highest cross-validated AUC (0.767) and the highest held-out AUC (0.707) of all image models (Fig.~\ref{fig:models}).

\begin{table}[!t]
\caption{Classifier Comparison (Image Features Only)}
\label{tab:models}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Model} & \textbf{CV AUC} & \textbf{Test AUC} & \textbf{Test Acc.} \\
\hline
\textbf{Soft-voting ensemble} & \textbf{0.767} & \textbf{0.707} & \textbf{66.7\%} \\
Logistic regression & 0.750 & 0.705 & 64.7\% \\
Gradient boosting & 0.744 & 0.699 & 64.2\% \\
Linear SVM & 0.757 & 0.699 & 64.5\% \\
RBF SVM & 0.766 & 0.696 & 65.7\% \\
MLP & 0.708 & 0.688 & 62.4\% \\
$k$-NN ($k=15$) & 0.727 & 0.673 & 65.1\% \\
Gaussian naive Bayes & 0.723 & 0.636 & 58.3\% \\
Random forest & 0.681 & 0.585 & 59.8\% \\
Decision tree & 0.516 & 0.512 & 37.5\% \\
\hline
\end{tabular}
\end{center}
\end{table}

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{01_model_comparison.png}}
\caption{Macro ROC-AUC per classifier: cross-validation against report-derived labels versus held-out test against radiologist labels.}
\label{fig:models}
\end{figure}

\subsection{Image, Report and Fused Performance}
Table~\ref{tab:headline} summarises the three information sources. Images and reports carry complementary evidence, so their fusion achieves the best performance on every metric: macro AUC 0.845, accuracy 80.6\% and NPV 88.5\%. Fig.~\ref{fig:sources} shows that the fused score is best or near-best for most findings.

\begin{table}[!t]
\caption{Held-out Performance by Information Source (58 Studies)}
\label{tab:headline}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{Image} & \textbf{Report} & \textbf{Fused} \\
\hline
Macro AUC & 0.707 & 0.808 & \textbf{0.845} \\
95\% CI & 0.653--0.757 & -- & 0.805--0.881 \\
Accuracy & 66.7\% & 80.3\% & \textbf{80.6\%} \\
Macro NPV & 81.1\% & 88.5\% & \textbf{88.5\%} \\
Macro recall & 65.4\% & 81.6\% & \textbf{81.6\%} \\
Macro F1 & 0.546 & 0.729 & \textbf{0.732} \\
\hline
\end{tabular}
\end{center}
\end{table}

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{06_sources.png}}
\caption{Held-out AUC per finding for the report reader, the image ensemble and their fusion.}
\label{fig:sources}
\end{figure}

\subsection{Per-Finding Results}
Table~\ref{tab:perlabel} reports fused results per finding. Ligament and meniscal tears are detected best (AUC 0.89--0.95). Negative predictions are highly reliable for ACL (NPV 96.9\%), MCL (97.8\%) and Baker's cyst (97.6\%), which is the property that matters most for a rule-out aid. Every finding exceeds an AUC of 0.73. Per-finding ROC curves are shown in Fig.~\ref{fig:roc}.

\begin{table}[!t]
\caption{Per-Finding Results, Image + Report Fusion}
\label{tab:perlabel}
\begin{center}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Finding} & \textbf{AUC} & \textbf{Acc.} & \textbf{Recall} & \textbf{NPV} \\
\hline
MCL & 0.948 & 91.4\% & 88.9\% & 97.8\% \\
Medial meniscus & 0.918 & 87.9\% & 84.6\% & 87.9\% \\
ACL & 0.913 & 93.1\% & 95.8\% & 96.9\% \\
Lateral meniscus & 0.894 & 86.2\% & 73.9\% & 84.6\% \\
Medial OA & 0.867 & 84.5\% & 93.3\% & 97.2\% \\
Baker's cyst & 0.864 & 87.9\% & 91.7\% & 97.6\% \\
Lateral OA & 0.843 & 79.3\% & 72.7\% & 92.7\% \\
Contusion & 0.837 & 69.0\% & 89.5\% & 92.0\% \\
PF OA & 0.794 & 75.9\% & 85.7\% & 89.7\% \\
Effusion & 0.788 & 69.0\% & 94.3\% & 77.8\% \\
Synovitis & 0.737 & 69.0\% & 48.1\% & 65.9\% \\
Fracture & 0.733 & 74.1\% & 61.1\% & 82.1\% \\
\hline
\textbf{Macro mean} & \textbf{0.845} & \textbf{80.6\%} & \textbf{81.6\%} & \textbf{88.5\%} \\
\hline
\end{tabular}
\end{center}
\end{table}

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{03_roc_curves.png}}
\caption{ROC curves per finding; panel titles give AUC for image only / image + report.}
\label{fig:roc}
\end{figure}

\subsection{Bias--Variance and Data Efficiency}
The validation curve over the logistic-regression regularisation parameter $C$ shows under-fitting for very small $C$ and a widening train--validation gap for larger $C$; the selected $C=10^{-3}$ lies at the validation peak. The learning curve rises steadily from a validation AUC of 0.670 with 173 studies to 0.756 with 3,479 studies and continues to climb, showing that KneeML scales with data (Fig.~\ref{fig:bv}). Balanced class weights raise logistic-regression recall from 30.2\% to 73.7\% at a fixed threshold without changing AUC.

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{05_bias_variance.png}}
\caption{Validation curve over $C$ (left) and learning curve over training-set size (right).}
\label{fig:bv}
\end{figure}

\subsection{Unsupervised Structure and Association Rules}
PCA requires 167 components to explain 80\% of the variance. $k$-means selects $k=4$ (silhouette 0.153), and $k$-medoids, Ward clustering and DBSCAN find consistent partitions, and these partitions follow acquisition protocol (adjusted Rand index 0.011 against an osteoarthritis grouping), confirming the value of KneeML's supervised, report-driven design. Apriori recovers 79 rules consistent with clinical knowledge, for example Fracture $\Rightarrow$ Contusion (confidence 0.726, lift 2.86) and Medial OA $\wedge$ PF OA $\Rightarrow$ Lateral OA (lift 2.74).

\subsection{End-to-End System Performance}
Running the full test set through the deployed web pipeline (CPU re-embedding, ensemble and fusion) yields 80.3\% accuracy over 696 decisions, reproducing the offline evaluation. Five studies are fully correct, more than half have at least 10 of 12 findings correct, and the median latency is 1.9~s per study on a laptop CPU (Table~\ref{tab:e2e}).

\begin{table}[!t]
\caption{End-to-End Web Pipeline on the Held-out Set}
\label{tab:e2e}
\begin{center}
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Decisions (studies $\times$ 12) & 696 \\
Accuracy, image + report & 80.3\% \\
Studies with 12/12 correct & 5 \\
Studies with $\ge$10/12 correct & 31 \\
Median latency (CPU) & 1.9~s \\
\hline
\end{tabular}
\end{center}
\end{table}

\subsection{Case Studies}
\textit{Acute injury with a non-English report (study K053).} The Turkish report describes a radial medial meniscal tear and discontinuity suggesting partial ACL rupture. The image model assigns 17\% to ACL; fusion raises it to 58\%, above threshold, and all 12 findings are correct (images alone: 9/12).

\textit{Complex degenerative knee (K052).} Eight concurrent findings (both menisci, three compartments of osteoarthritis, effusion, synovitis, Baker's cyst) are all detected with no false positives.

\textit{Combined ligament injury (K004).} Fusion raises the ACL, MCL and lateral meniscal tear probabilities from 15--25\% to 58--62\%, lifting the study from 5/12 to 11/12 and illustrating the complementarity of the two sources.

\textit{Nine concurrent findings from images alone (K027).} For a knee with ACL and bilateral meniscal tears, two-compartment osteoarthritis, effusion, synovitis and a Baker's cyst, the image ensemble alone scores 11/12 (effusion 0.94, PF OA 0.89, Baker's cyst 0.76), and fusion further raises confidence (effusion 0.97, PF OA 0.95).

\section{Discussion}

Report-driven supervision turns an entire multi-centre archive into training data without manual annotation, and a frozen encoder with classical heads makes the full system trainable in minutes and auditable at every stage: the label rules are readable, each model is a standard estimator, thresholds are set per finding, and the web interface explains each call in plain language. Images and reports provide complementary evidence, and their fusion delivers a macro AUC of 0.845 with ligament and meniscal detection at 0.89--0.95 and highly reliable negative calls. The steadily rising learning curve indicates that KneeML continues to improve as more studies are added, and the lightweight design means it runs in under two seconds per study on commodity hardware.

\section{Future Work}

KneeML provides a strong foundation for several extensions. A learned text classifier trained on the rule outputs, as CheXbert did for CheXpert \cite{b9}, can further enrich the labels. A medical-domain or MRI fine-tuned encoder, more slices and 3D or attention-based pooling can extend the image branch, and the rising learning curve shows that additional studies will translate directly into gains. An expanded radiologist-verified set will enable active learning and a learned per-finding fusion weight. Finally, slice- and region-level explanations and a prospective reader study will bring KneeML closer to clinical deployment as a second reader.

\section{Conclusion}
\label{sec:conclusion}

KneeML shows that an accurate 12-finding knee MRI classifier can be built directly from a large clinical archive, by mining multilingual reports for training labels, describing images with a frozen self-supervised encoder, and learning with calibrated classical classifiers. On an independent radiologist-verified test set never used during development, image and report fusion achieves a macro AUC of 0.845, 80.6\% accuracy and 88.5\% NPV, with ligament and meniscal tears detected at AUCs of 0.89--0.95, while requiring one short GPU job and running in under two seconds per study on a CPU. The approach is inexpensive, transparent and reproducible, and offers a practical starting point wherever reports are plentiful and expert labels are not.

\section*{Acknowledgment}

We extend our sincere appreciation to BRACT's Vishwakarma Institute of Technology and our Director, Prof. Dr. Rajesh Jalnekar, for their support throughout this project, and to the providers of the public de-identified knee MRI collection used in this work.

\begin{thebibliography}{00}

\bibitem{b1} N. Bien \textit{et al.}, ``Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet,'' \textit{PLOS Medicine}, vol.~15, no.~11, e1002699, 2018.

\bibitem{b2} C.-H. Tsai, N. Kiryati, E. Konen, I. Eshed, and A. Mayer, ``Knee injury detection using MRI with efficiently-layered network (ELNet),'' in \textit{Proc. Medical Imaging with Deep Learning (MIDL)}, PMLR 121, 2020, pp. 784--794.

\bibitem{b3} D. Azcona, K. McGuinness, and A. F. Smeaton, ``A comparative study of existing and new deep learning methods for detecting knee injuries using the MRNet dataset,'' \textit{arXiv preprint arXiv:2010.01947}, 2020.

\bibitem{b4} F. Liu \textit{et al.}, ``Deep learning approach for evaluating knee MR images: Achieving high diagnostic performance for cartilage lesion detection,'' \textit{Radiology}, vol.~289, no.~1, pp. 160--169, 2018.

\bibitem{b5} V. Pedoia \textit{et al.}, ``3D convolutional neural networks for detection and severity staging of meniscus and PFJ cartilage morphological degenerative changes in osteoarthritis and anterior cruciate ligament subjects,'' \textit{J. Magn. Reson. Imaging}, vol.~49, no.~2, pp. 400--410, 2019.

\bibitem{b6} X. Wang \textit{et al.}, ``ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,'' in \textit{Proc. IEEE CVPR}, 2017, pp. 2097--2106.

\bibitem{b7} J. Irvin \textit{et al.}, ``CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,'' in \textit{Proc. AAAI}, vol.~33, 2019, pp. 590--597.

\bibitem{b8} W. W. Chapman \textit{et al.}, ``A simple algorithm for identifying negated findings and diseases in discharge summaries,'' \textit{J. Biomed. Inform.}, vol.~34, no.~5, pp. 301--310, 2001.

\bibitem{b9} A. Smit \textit{et al.}, ``CheXbert: Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT,'' in \textit{Proc. EMNLP}, 2020, pp. 1500--1519.

\bibitem{b10} A. Ratner \textit{et al.}, ``Snorkel: Rapid training data creation with weak supervision,'' \textit{Proc. VLDB Endowment}, vol.~11, no.~3, pp. 269--282, 2017.

\bibitem{b11} M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, ``Transfusion: Understanding transfer learning for medical imaging,'' in \textit{Proc. NeurIPS}, 2019.

\bibitem{b12} M. Oquab \textit{et al.}, ``DINOv2: Learning robust visual features without supervision,'' \textit{Trans. Mach. Learn. Res.}, 2024.

\bibitem{b13} C. Cortes and V. Vapnik, ``Support-vector networks,'' \textit{Machine Learning}, vol.~20, no.~3, pp. 273--297, 1995.

\bibitem{b14} L. Breiman, ``Random forests,'' \textit{Machine Learning}, vol.~45, no.~1, pp. 5--32, 2001.

\bibitem{b15} J. Platt, ``Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,'' in \textit{Advances in Large Margin Classifiers}. MIT Press, 1999, pp. 61--74.

\bibitem{b16} F. Pedregosa \textit{et al.}, ``Scikit-learn: Machine learning in Python,'' \textit{J. Mach. Learn. Res.}, vol.~12, pp. 2825--2830, 2011.

\bibitem{b17} R. Agrawal and R. Srikant, ``Fast algorithms for mining association rules,'' in \textit{Proc. VLDB}, 1994, pp. 487--499.

\bibitem{b18} M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, ``A density-based algorithm for discovering clusters in large spatial databases with noise,'' in \textit{Proc. KDD}, 1996, pp. 226--231.

\end{thebibliography}
\end{document}
