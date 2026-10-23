# How we built KneeML (in plain words)

## By the numbers

| | |
|---:|---|
| **~250 GB** | of knee MRI data processed |
| **4,410** | knee scans, from **24,386** picture stacks |
| **105,840** | MRI pictures read by an AI vision model |
| **81 million** | numbers extracted from those pictures |
| **10+ languages** | of doctors' reports read automatically |
| **52,188** | answers generated from reports, with **zero** hand-labelling |
| **300+** | hand-written language rules in the report reader |
| **9** | different machine-learning methods compared |
| **1,700+** | individual classifiers trained and tested |
| **0** | failures across all 4,410 scans |
| **~35 min** | of free Kaggle graphics-card time, the only heavy computing needed |
| **~2 seconds** | to analyse a new scan on an ordinary laptop |
| **~3,500** | lines of code, from raw scans to a working website |

## Headline results

On scans the system had **never seen**, checked against **expert radiologists**:

- **0.948**: MCL tear detection score (1.0 = perfect, 0.5 = guessing)
- **0.918**: medial meniscus tear detection score
- **0.913**: ACL tear detection score
- **93%**: ACL answers correct · **91%** MCL answers correct
- **97–98%**: of "all clear" calls correct for ACL, MCL and Baker's cyst
- **88.5%**: of all "all clear" calls correct, across the 12 problems
- **80.6%**: of all **696** yes/no answers correct (pictures + report)
- **53** separate measurements at or above **85%**
- **5** test scans got **all 12 answers right**; **31** got at least 10 of 12 right
- **79** medical patterns discovered automatically, such as "**3 in 4** broken bones come with a
  bone bruise" (**2.9×** more often than chance)

## What it does

You give it a knee MRI scan. It tells you, for 12 common knee problems, how likely each one is:

- torn ligaments (ACL, MCL)
- torn cartilage cushions inside the knee (medial and lateral meniscus)
- wear-and-tear arthritis in three parts of the knee
- fluid in the joint, inflamed joint lining, a fluid-filled lump behind the knee (Baker's cyst)
- bruised bone, broken bone

A knee can have several of these at once, so the computer answers 12 separate yes/no questions.

## What we started with

- **4,410 knee scans.** Each scan is several stacks of pictures taken from three directions:
  side-on, front-on and top-down. That's about 24,000 picture stacks and roughly a quarter of a
  terabyte of data.
- **A written doctor's report for every scan**, in more than ten languages (Spanish, English,
  German, Dutch, Turkish, Greek, Bulgarian and others).
- **Only 58 scans where an expert had filled in the 12 answers.**

The big challenge: a computer learns from examples with answers, and we had only 58 answered
examples. That's far too few.

## Step 1: get answers from the reports

Every scan already has a report where a doctor wrote what they saw, for example "complete tear of
the ACL" or "no fluid in the joint". So we wrote a set of **reading rules** that scan each report
and pull out the 12 answers:

- look for a body part ("ACL", "meniscus", in every language) next to a problem word ("tear",
  "rotura", "Riss", "scheur", ...)
- watch for "no", "without", "normal", "intact" (and their translations), which flip the meaning
- treat "possible" or "suspected" as only half-sure
- ignore the question the referring doctor asked ("rule out ACL tear?"), because it isn't a finding

This gave us answers for **4,349 scans**, no hand-labelling needed. We checked the rules against
the 58 expert-answered scans: they agree about **80% of the time**. That's not perfect, because
doctors sometimes see things in the pictures they don't write down, but it's good enough to learn
from.

## Step 2: turn each scan into numbers

A computer can't compare pictures directly, so each scan has to become a list of numbers that
describes it.

1. From each scan we took the best picture stack from each of the three directions (the kind
   where fluid and swelling show up brightly).
2. From each stack we took **8 evenly spaced pictures** from the middle part, 24 pictures per
   scan in total.
3. We evened out the brightness, made every picture the same size, and flipped left knees so
   they look like right knees. That way "inner" and "outer" are always on the same side.
4. We fed each picture to a ready-made picture-understanding program (DINOv2). It was trained
   by others on millions of everyday photos, and we didn't change it. For each picture it gives
   back **768 numbers** that describe what's in it.
5. We combined the 24 pictures' numbers into **one list of about 4,600 numbers per scan**.

This step needed a lot of computing power, so it ran on Kaggle's free graphics cards. It took
about 35 minutes for all 4,410 scans, with no failures.

## Step 3: teach the computer

Now every scan is a list of numbers plus 12 yes/no answers from the reports. We tried
**nine different classic learning methods** from our course, each one learning the link between
the numbers and the answers:

| Method | The idea in one line |
|---|---|
| Logistic regression | draws the best straight dividing line between "yes" and "no" scans |
| Naive Bayes | uses probability: "how often do scans with these numbers have a tear?" |
| k-Nearest Neighbours | looks at the most similar past scans and copies their answers |
| Linear SVM | like a dividing line, but pushed as far as possible from both groups |
| RBF SVM | the same, but the dividing line can curve |
| Decision tree | asks a chain of simple questions about the numbers |
| Random Forest | hundreds of decision trees that vote |
| Gradient boosting | trees added one at a time, each fixing the last one's mistakes |
| Neural network | a small network of connected "neurons" |

Before learning, we squeezed the 4,600 numbers down to the most informative 64–128 summary numbers
(this is called PCA). That keeps the methods fast and stops them memorising noise.

**Fair testing:**
- The 58 expert-answered scans were locked away and **never used for learning or tuning**.
  They are the final exam.
- For practice tests while tuning, we split the other 4,349 scans into 5 groups: learn on 4, test
  on the 5th, and rotate (this is called cross-validation). Scans that share one report were kept
  in the same group, so nothing leaks between practice and test.
- We tried several settings for each method and kept whichever did best on the practice tests.

**Combining:** the three best methods (RBF SVM, linear SVM and logistic regression) vote together.
Each one's raw score is first converted into a proper 0–100% probability, then we average them.

**Using the report too:** when a report is available, we average the picture-based answer with
the report-reading answer. The two sources make different mistakes, so together they do better.

## Results (on the 58 locked-away expert scans)

| | Pictures + report | Pictures only | Report rules only |
|---|---:|---:|---:|
| Answers correct (all 12 problems) | **80.6%** | 66.7% | 80.3% |
| "Says no and is right" rate (NPV) | **88.5%** | 81.1% | – |
| Ranking score (AUC, 0.5 = guessing, 1 = perfect) | **0.845** | 0.707 | 0.808 |

Some problems are found much better than others (pictures + report):

- ACL tear: **93%** of answers correct, ranking score 0.91
- MCL tear: **91%** correct, ranking score 0.95
- Medial meniscus tear: ranking score 0.92
- When the system says there is **no** ACL tear, no MCL tear or no Baker's cyst, it is right
  **97–98%** of the time (and from pictures alone, a "no fluid" answer was right every time)

Running the actual website on all 58 scans gives the same picture: **80.3%** of the 696 answers
correct, 31 scans with at least 10 of 12 right, 5 scans perfect, about **2 seconds per scan** on a
normal laptop.

Honest caveats:
- 58 test scans is small, so these numbers could easily move by a few points with a different set.
- The picture-only system is weaker because it learned from report-based answers, which are only
  about 80% right themselves. The computer can't learn to be much better than its teacher.
- It is a student project, **not a medical tool**.

## Extra things we looked at

- **Grouping without answers** (K-means, K-medoids, hierarchical clustering, DBSCAN): when we let
  the computer group the scans by itself, it grouped them mostly by **which scanner and settings
  were used**, not by disease. That shows why the answers (Step 1) were needed.
- **Which problems come together** (Apriori): for example, a broken bone usually comes with a
  bone bruise (about 3× more often than chance), and arthritis in one part of the knee usually
  means arthritis in the others.
- **Too simple vs too complex** (bias and variance): we showed that a model that's too
  restricted misses patterns, and one that's too free memorises the practice data and does
  worse on new scans.
- **Rare problems** (class imbalance): some problems appear in only 6% of scans. We told the
  methods to pay extra attention to the rare "yes" cases, and we set a separate yes/no cut-off
  for each problem.

## The website

- **Inference:** pick one of the 58 test scans, scroll through its pictures, press Predict, and
  see the 12 probabilities next to the expert's answers. You can also upload your own pictures.
- **Results:** every number and chart from this project.
- **Method:** this process in short, plus which course topics it uses.

It runs on your own computer: start it with
`python -m uvicorn backend.app:app --port 8010` and open http://localhost:8010.

## Where things live

| Folder or file | What's in it |
|---|---|
| `kneeml/` | all the Python code (report reading, learning, testing, charts) |
| `kaggle/` | the script that turned the scans into numbers on Kaggle |
| `backend/`, `frontend/` | the website |
| `results/` | all the numbers and charts |
| `inference_samples/` | 10 test scans ready to upload on the website |
| `technical.md` | the full technical version of this document |
| `pitch.md` | a one-page pitch |
