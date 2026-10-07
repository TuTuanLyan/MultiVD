# Label correctness and duplication in software vulnerability detection (VD) benchmark datasets

Status legend used on every item: **VERIFIED** = number/quote read in the primary-source full text (arXiv PDF converted with `pdftotext`, or publisher/Crossref metadata for bibliographic fields); **PARTIAL** = abstract or metadata only; **UNVERIFIED** = secondary source only. All arXiv PDFs were fetched on 2026-09-28 from the exact URLs cited. Bibliographic fields (pages, DOI, venue) were checked against the Crossref REST API (`https://api.crossref.org/works/<DOI>`) unless noted.

## Q1. Does any primary source report "about 22%" label correctness for Devign / Big-Vul-type datasets?

### Takeaway
No primary source I read reports a label-correctness figure of 22% for Devign or Big-Vul. The closest verified figures are **24.0%** for CodeXGLUE (the Devign FFmpeg+QEMU release) in PrimeVul (ICSE 2025) and **25%** for BigVul in DiverseVul (RAID 2023), which PrimeVul reuses. Croft et al.'s "20%" is an *inaccuracy* rate for Devign, which means 80% of its labels were correct. A "22" does appear in three other places, but none of them measures label correctness: 22.10% duplicates (ReVeal), an MCC of 0.22 (Jimenez et al.), and "22 papers" (Top Score).

### Cited Findings
Consolidated table of **label-correctness (share of vulnerable-labelled functions that are really vulnerable)** from primary sources:

| Dataset | Correct | Sample audited | Source (location) | Status |
|---|---|---|---|---|
| CodeXGLUE / Devign (FFmpeg+QEMU) | **24.0%** | 50 vulnerable fns | PrimeVul, Table I + Suppl. Table IX | VERIFIED |
| Devign | 80.0% (i.e. 20% inaccurate) | 70 random samples | Croft et al. ICSE'23, Table III | VERIFIED |
| Devign | 47% (arXiv v2 / ISSTA'25); 50% (arXiv v1) | 100 vulnerable fns | Top Score on the Wrong Exam, Fig. 6 | VERIFIED |
| BigVul | **25%** | part of a 50-fn union sample | DiverseVul, Table 8 (reused as 25.0* in PrimeVul Table I) | VERIFIED |
| Big-Vul | 54.3% (45.7% inaccurate) | 70 | Croft et al., Table III | VERIFIED |
| BigVul | 39% (v2) / 38% (v1) | 100 | Top Score, Fig. 6 | VERIFIED |
| CVEFixes ∪ BigVul ∪ CrossVul | 36% | 50 | DiverseVul, Table 8 | VERIFIED |
| CrossVul | 47.8% | part of 50 | DiverseVul, Table 8 | VERIFIED |
| CVEFixes | 51.7% | part of 50 | DiverseVul, Table 8 | VERIFIED |
| DiverseVul | 60% | 50 | DiverseVul, Table 8 | VERIFIED |
| DiverseVul | 65% (v2) / 64% (v1) | 100 | Top Score, Fig. 6 | VERIFIED |
| VulnPatchPairs | 36% | 50 | PrimeVul, Table I | VERIFIED |
| D2A | 28.6% | 70 | Croft et al., Table III | VERIFIED |
| D2A (own check, biased sample) | 53% with auto-labeler, 35% without | 57 (41 pos / 16 neg) | D2A paper, Sec. "Manual Label Validation" | VERIFIED |
| SVEN | 94.0% | 50 | PrimeVul, Table I | VERIFIED |
| PrimeVul-OneFunc / NVDCheck | 86.0% / 92.0% | 50 each | PrimeVul, Table I | VERIFIED |
| CleanVul (threshold 3) | 90.6% (uncleaned GitHub VFCs: 28.7%) | 487 function-level changes | CleanVul RQ1, Table 3 | VERIFIED |
| ReposVul | 85% C / 90% C++ / 85% Java / 80% Python | 50 CVE cases per language | ReposVul, Table 5 | VERIFIED |
| Juliet (synthetic) | 100% | 70 | Croft et al., Table III | VERIFIED |

- PrimeVul, verbatim: "Surprisingly, the widely used dataset CodeXGLUE [8] (a.k.a. Devign [19] dataset) has a label accuracy of only 24% for vulnerable functions, even though Zhou et al. [19] had recruited human annotators to label security-related commits." Section II-B.2 - [PrimeVul arXiv v2](https://arxiv.org/pdf/2403.18624) (VERIFIED)
- DiverseVul, verbatim: "Within these three datasets, CVEFixes is the most accurate one, whereas BigVul has very low label accuracy, only 25%." Section 5 (label noise), Table 8 - [DiverseVul arXiv v2](https://arxiv.org/pdf/2304.00409) (VERIFIED)
- Croft et al., verbatim: "We obtained accuracy values of 0.8 (Devign), 0.543 (Big-Vul), and 0.286 (D2A)." Section IV-A - [Croft et al. arXiv](https://arxiv.org/pdf/2301.05456) (VERIFIED)
- Non-label "22" coincidences:
  - ReVeal: "SySeVR's preprocessing technique introduces 25.56% duplicates in REVEAL dataset and 22.10% duplicates in FFMPeg+Qemu". This is **duplication after slicing and tokenisation**, not label correctness. [ReVeal arXiv](https://arxiv.org/pdf/2009.07235) (VERIFIED)
  - Jimenez et al. FSE'19: under realistic labelling, "MCC mean values ... drop from 0.77, 0.65 and 0.43 to 0.08, 0.22, 0.10 for Linux Kernel, OpenSSL and Wiresark". This is a **model MCC**. [Semantic Scholar record](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3338906.3338941?fields=title,abstract) (PARTIAL, abstract only)
  - Top Score: "BigVul [29] and Devign [30] were both used by 36 papers, and ReVeal [31] by 22 papers." [Top Score arXiv v2](https://arxiv.org/pdf/2408.12986) (VERIFIED)

### Inferences
- The remembered "about 22%" is most plausibly PrimeVul's **24%** for CodeXGLUE/Devign, or DiverseVul/PrimeVul's **25%** for BigVul. A third possibility is Croft's "20%" (the lower end of "20-71% inaccurate") read with its sign flipped. If a paper needs a number "near 22%", cite **24% (CodeXGLUE/Devign, PrimeVul, n=50)** and **25% (BigVul, DiverseVul)** exactly. Do not write 22%.
- Estimates for the same dataset differ by a factor of up to 3.3 (Devign: 80% Croft, 50%/47% Top Score, 24% PrimeVul). The main driver is the labelling criterion, and the papers say so themselves:
  - Croft counts callers of vulnerable functions, and ambiguous cases, as correct.
  - DiverseVul, PrimeVul and Top Score count callers, and functions holding only part of a multi-function vulnerability, as wrong.
- Every one of these audits is small (50-100 functions, and fewer per dataset in DiverseVul's union). At n=50 a 95% binomial interval is roughly ±12-14 points, so 24% and 25% are not distinguishable from each other. This is my calculation, not a figure from the papers.

### Gaps
- I did not read the ICSE 2025 camera-ready of PrimeVul (IEEE Xplore). Its arXiv v1 and v2 give identical Table I/II numbers.
- No primary source found reporting exactly 22% label correctness. Searches for "22%" with Devign/BigVul label terms surfaced only the sources above.

## Q2. Croft, Babar, Kholoosi, "Data Quality for Software Vulnerability Datasets" (ICSE 2023): per-dataset breakdown

### Takeaway
The per-dataset values are in **Table III** of the arXiv full text (only one arXiv version, v1, 13 Jan 2023). Accuracy is Big-Vul 0.543, Devign 0.800, D2A 0.286, Juliet 1.000. Uniqueness is 0.830 / 0.899 / 0.021 / 0.163 in the same order. The abstract's "20-71%" inaccuracy is 1-0.800 and 1-0.286; the "17%" end of "17-99% duplicated" is 1-0.830.

### Cited Findings
- **Bibliographic record** (VERIFIED, Crossref + arXiv PDF): Roland Croft, M. Ali Babar, M. Mehdi Kholoosi. "Data Quality for Software Vulnerability Datasets." *2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)*, pp. 121-133, May 2023. DOI 10.1109/ICSE48619.2023.00022. arXiv:2301.05456 (v1 only, 13 Jan 2023). URL read: https://arxiv.org/pdf/2301.05456 - [Crossref](https://api.crossref.org/works/10.1109/ICSE48619.2023.00022)
- **Table III "Measured value of each attribute for each dataset"** (VERIFIED; the note under the table reads "* Based on a sample of the data", which applies to Accuracy) - [arXiv](https://arxiv.org/pdf/2301.05456):

| Attribute | Big-Vul | Devign | D2A | Juliet |
|---|---|---|---|---|
| Accuracy* | 0.543 | 0.800 | 0.286 | 1.000 |
| Uniqueness | 0.830 | 0.899 | 0.021 | 0.163 |
| Consistency | 0.999 | 0.991 | 0.531 | 0.750 |
| Completeness | 0.824 | 0.944 | 0.981 | 1.000 |
| Currentness | 0.761 | 0.811 | 0.844 | - |

- **Table II** (dataset profile, VERIFIED):
  - Big-Vul: "Security vendor provided", 188,636 functions, 5.78% vulnerable.
  - Devign: "Developer provided", 27,318 functions, 45.61% vulnerable.
  - D2A: "Tool created", 1,295,623 functions, 1.44% vulnerable.
  - Juliet: "Synthetically created", 253,002 functions, 36.77% vulnerable.
  - [arXiv](https://arxiv.org/pdf/2301.05456)
- **Accuracy method** (VERIFIED, Sec. IV-A):
  - Only the vulnerable class was audited: "our analysis was constrained to the vulnerable label source. We focused our investigation on label correctness of data points labelled as vulnerable."
  - Sample and raters: "we examined 70 random samples of each dataset (90% confidence level +/- 10%)". Two authors rated independently, with "a Cohen Kappa value of 0.627".
  - Ambiguous cases were counted as correct: "We tentatively labeled these ambiguous cases as correct."
  - Juliet: "We found no inaccuracies within the synthetic Juliet dataset, as the vulnerable cases are crafted specifically for the label".
  - [arXiv](https://arxiv.org/pdf/2301.05456)
- **Table IV "Types of label inaccuracy in real-world datasets"** (VERIFIED):
  - Big-Vul: irrelevant 25%, cleanup 28.1%, inaccurate fix identification 46.9%.
  - Devign: 42.9% / 21.4% / 35.7%.
  - D2A: 0 / 0 / 100%.
  - Text: "Over two-thirds of the D2A labels were inaccurate"; "Most errors [Big-Vul] arose from inaccuracies in tracing the fixing commits, particularly for the Chromium project. 36% of the vulnerable entries in Big-Vul are from the Chromium project."
  - [arXiv](https://arxiv.org/pdf/2301.05456)
- **Impact of inaccurate labels** (VERIFIED): "The precision decreased by 29%, 50% and 80% for Devign, Big-Vul and D2A, respectively". Recall was not significantly affected. [arXiv](https://arxiv.org/pdf/2301.05456)
- **Uniqueness** (VERIFIED, Sec. IV-B):
  - Duplicates were type-3 clones found with Allamanis's duplicate detector: "We obtained a uniqueness value of 0.830 (Big-Vul), 0.899 (Devign), 0.021 (D2A), and 0.163 (Juliet)."
  - D2A: "Each unique function in the dataset had an average of 57 duplicates".
  - Intro: "A few datasets exhibited large data duplication rates, between 17-99%. ... Evaluation performance decreased by up to 82% after removing such duplicates."
  - Table V ("Performance impact of uniqueness issues") Change column: Big-Vul 0.0%, Devign 13.9%, D2A 81.7%, Juliet 10.4%.
  - [arXiv](https://arxiv.org/pdf/2301.05456)
- **Consistency** (VERIFIED): the text gives "consistency values for Big-Vul (0.999) and Devign (0.991), but lower values for D2A (0.531) and Juliet (0.75)", and the intro says "up to 47% of labels were inconsistent". [arXiv](https://arxiv.org/pdf/2301.05456)
- **Completeness** (VERIFIED): measured with an automatic C/C++ syntax check for truncated functions. "we observed completeness values of 0.824, 0.944, 0.981, and 1.0 for Big-Vul, Devign, D2A and Juliet". [arXiv](https://arxiv.org/pdf/2301.05456)
- **Currentness** (VERIFIED): computed as 1 minus the Jensen-Shannon divergence. "currentness values of 0.761 (Big-Vul), 0.811 (Devign) and 0.844 (D2A). These values are relatively high for this attribute and are unlikely to indicate concept drift." [arXiv](https://arxiv.org/pdf/2301.05456)

### Inferences
- The accuracies correspond to 56/70 correct (Devign), 38/70 (Big-Vul) and 20/70 (D2A). Big-Vul's 32/70 = **45.7%** inaccurate is the figure Top Score cites. This is my arithmetic from Table III.
- The "99%" upper end of "17-99% duplicated" does not follow exactly from Table III. The lowest uniqueness, 0.021 (D2A), implies 97.9% non-unique. It may reflect rounding or a different count, and the paper does not reconcile it. Cite the Table III values rather than "99%".
- Croft's Devign 0.800 is the most lenient estimate in the literature, because callers and ambiguous functions count as correct.

### Gaps
- I did not read the IEEE Xplore camera-ready. arXiv has only v1, so version differences cannot be checked.
- The Figure 3 / Table VI-VII detail on consistency and truncation types was only partly extracted.

## Q3. DiverseVul (Chen, Ding, Alowain, Chen, Wagner; RAID 2023): manual label check

### Takeaway
Your notes are confirmed: DiverseVul 60%, BigVul 25%, CrossVul 47.8%, CVEFixes 51.7%, and the union of the three prior datasets 36%. The audit sampled **50** vulnerable functions from DiverseVul and **50** from the union CVEFixes ∪ BigVul ∪ CrossVul, so the per-dataset figures come from sub-samples of that union. The analysis first appears in **arXiv v2 (9 Aug 2023)** and is absent from v1 (1 Apr 2023).

### Cited Findings
- **Bibliographic record** (VERIFIED, Crossref + arXiv PDF): Yizheng Chen, Zhoujie Ding, Lamya Alowain, Xinyun Chen, David Wagner. "DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection." *Proceedings of the 26th International Symposium on Research in Attacks, Intrusions and Defenses (RAID '23)*, Hong Kong, Oct 16-18 2023, ACM, pp. 654-668. DOI 10.1145/3607199.3607242. arXiv:2304.00409 (v1 1 Apr 2023; v2 9 Aug 2023). URL read: https://arxiv.org/pdf/2304.00409 - [Crossref](https://api.crossref.org/works/10.1145/3607199.3607242)
- **Dataset size** (VERIFIED): "18,945 vulnerable functions spanning 150 CWEs and 330,492 non-vulnerable functions extracted from 7,514 commits". [arXiv v2](https://arxiv.org/pdf/2304.00409)
- **Method** (VERIFIED):
  - Sampling: "We randomly sample 50 vulnerable functions from DiverseVul, and 50 vulnerable functions from the union of previous three datasets (CVEFixes ∪ BigVul ∪ CrossVul)."
  - Evidence used: the function before and after the commit, the commit, the CVE description and the developer discussion.
  - Criterion: "We confirm a function as correctly labelled vulnerable if the vulnerability exists in that function, and is not spread across multiple functions."
  - [arXiv v2](https://arxiv.org/pdf/2304.00409)
- **Table 8 "Label accuracy of four datasets, evaluated on a random sample of vulnerable functions"** (VERIFIED). The three right-hand columns are the Wrong Label breakdown. [arXiv v2](https://arxiv.org/pdf/2304.00409)

| Dataset | Correct Label | Spread across multiple functions | Relevant consistency | Irrelevant |
|---|---|---|---|---|
| DiverseVul | 60% | 10% | 12% | 18% |
| CVEFixes ∪ BigVul ∪ CrossVul | 36% | 12% | 12% | 40% |
| CVEFixes | 51.7% | 10.3% | 17.3% | 20.7% |
| BigVul | 25% | 15.6% | 9.4% | 50% |
| CrossVul | 47.8% | 13% | 21.8% | 17.4% |

- **Verbatim wording** (VERIFIED):
  - "The vulnerable function labels are 60% accurate in DiverseVul, which is 24 percentage points higher than the previous three datasets (CVEFixes ∪ BigVul ∪ CrossVul). Within these three datasets, CVEFixes is the most accurate one, whereas BigVul has very low label accuracy, only 25%. We observe that many commits included in BigVul from the Chromium and Android projects are not relevant to fixing vulnerabilities at all."
  - "the percentage of irrelevant functions is surprisingly high, ranging from 17.4% to 50% in four datasets."
  - [arXiv v2](https://arxiv.org/pdf/2304.00409)
- **Stricter criterion than Croft** (VERIFIED): "we consider the caller of a vulnerable function as non-vulnerable; they considered it vulnerable. Also, if a function is only part of the vulnerability ... we consider that a wrong label; they considered it correct." [arXiv v2](https://arxiv.org/pdf/2304.00409)
- **Whitespace-only label errors and dedup** (VERIFIED, Sec. 6 Limitations):
  - "we discovered that 4% of DiverseVul labels and 6% of (CVEFixes ∪ BigVul ∪ CrossVul) labels were erroneous because the commit made whitespace-only changes to some functions".
  - Deduplication is by MD5 of the function text.
  - [arXiv v2](https://arxiv.org/pdf/2304.00409)
- **Version difference** (VERIFIED): arXiv v1 (1 Apr 2023) has no manual label-accuracy table. Its "Table 8" is about class weights, and it says only that "some label noise may be present for vulnerable functions". [arXiv v1](https://arxiv.org/pdf/2304.00409v1)

### Inferences
- The per-dataset percentages fit these implied sub-sample sizes:
  - BigVul n≈32 (8, 5, 3 and 16 functions).
  - CVEFixes n≈29 (15, 3, 5, 6).
  - CrossVul n≈23 (11, 3, 5, 4).

  These sum to more than 50, so a sampled function can belong to several datasets. The paper does not state these n; they are my back-calculation. The BigVul 25% therefore rests on roughly 32 functions.
- PrimeVul's Table I marks the BigVul/CrossVul/CVEFixes/DiverseVul values with "*", meaning they were reused from this paper, not re-measured. The 25% / 47.8% / 51.7% / 60% therefore come from a single audit.

### Gaps
- I did not check the ACM DL camera-ready separately. The arXiv v2 date (9 Aug 2023) is close to RAID (Oct 2023), and its header carries the RAID '23 ACM reference.

## Q4. PrimeVul (Ding et al., ICSE 2025): label accuracy of existing datasets, PrimeVul's own labels, and duplication

### Takeaway
PrimeVul audited 50 vulnerable functions per benchmark with 3 annotators and majority vote. Results: SVEN **94.0%**, CodeXGLUE/Devign **24.0%**, VulnPatchPairs **36.0%**, PrimeVul-OneFunc **86.0%**, PrimeVul-NVDCheck **92.0%**. BigVul/CrossVul/CVEFixes/DiverseVul are copied from DiverseVul. The exact-copy rate of test vulnerable functions found in training is CVEFixes 18.9%, BigVul 12.7%, DiverseVul 3.3%, CodeXGLUE 0.6%, PrimeVul 0.0%. arXiv v1 and v2 give identical numbers.

### Cited Findings
- **Bibliographic record** (VERIFIED, Crossref + arXiv PDF): Yangruibo Ding, Yanjun Fu, Omniyyah Ibrahim, Chawin Sitawarin, Xinyun Chen, Basel Alomair, David Wagner, Baishakhi Ray, Yizheng Chen. "Vulnerability Detection with Code Language Models: How Far Are We?" *2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)*, pp. 1729-1741, 2025. DOI 10.1109/ICSE55347.2025.00038. arXiv:2403.18624 (v1 27 Mar 2024; v2 10 Jul 2024). URL read: https://arxiv.org/pdf/2403.18624 and https://arxiv.org/pdf/2403.18624v1 - [Crossref](https://api.crossref.org/works/10.1109/ICSE55347.2025.00038)
- **Method** (VERIFIED, Sec. II-B.1):
  - Sampling: "randomly sampling 50 vulnerable functions from each benchmark and manually analyzing whether the function indeed contains security vulnerabilities."
  - Annotators: three, "including two researchers with several years of experience in computer security and one senior security expert".
  - Vote: "we use three human annotators with majority vote labeling for all datasets except SVEN."
  - [arXiv v2](https://arxiv.org/pdf/2403.18624)
- **Table I "The label accuracy across existing vulnerability benchmarks..."** (VERIFIED; identical in v1 and v2). Values marked * are "label accuracy numbers in Chen et al. [5]". [arXiv v2](https://arxiv.org/pdf/2403.18624)

| Benchmark | Manual? | Correct (%) |
|---|---|---|
| SVEN | yes | 94.0 |
| CodeXGLUE | yes | 24.0 |
| VulnPatchPairs | yes (†) | 36.0 |
| BigVul | no | 25.0* |
| CrossVul | no | 47.8* |
| CVEFixes | no | 51.7* |
| DiverseVul | no | 60.0* |
| PrimeVul-OneFunc | no | 86.0 |
| PrimeVul-NVDCheck | no | 92.0 |

  † "The VulnPatchPairs dataset takes pairs of functions from CodeXGLUE. The dataset does not involve further manual verification beyond its data resource, CodeXGLUE."
- **Wording** (VERIFIED):
  - "the benchmarks without manual verification have very low accuracy between 25% and 60%, for vulnerable functions."
  - "SVEN has a 94% label accuracy for vulnerable functions."
  - "We find that their human annotation on security-related commits is highly inaccurate, such that many commits in CodeXGLUE do not fix security vulnerabilities."
  - [arXiv v2](https://arxiv.org/pdf/2403.18624)
- **Supplementary Table IX "Detailed breakdown for label error analysis"** (VERIFIED). The last three columns are wrong-label categories. [arXiv v2, Supplementary Material A](https://arxiv.org/pdf/2403.18624)

| Benchmark | Correct | Spread across functions | Relevant consistency | Irrelevant |
|---|---|---|---|---|
| SVEN | 94% | 0% | 0% | 6% |
| CodeXGLUE | 24% | 18% | 0% | 58% |
| VulnPatchPairs | 36% | 10% | 14% | 40% |
| PrimeVul-OneFunc | 86% | 4% | 4% | 6% |
| PrimeVul-NVDCheck | 92% | 4% | 2% | 2% |

  Text: "Contrasting our findings, Croft et al. [50] acknowledge a labeling accuracy issue with CodeXGLUE, yet their review finds 80% of their sampled vulnerable functions correctly labeled. This divergence largely stems from our more stringent criteria".
- **Table II "The statistics of data duplication in existing vulnerability detection benchmarks"** (VERIFIED; identical in v1 and v2). [arXiv v2](https://arxiv.org/pdf/2403.18624)
  - Method: "We study the exact copy of vulnerable functions ... we exhaustively compare the vulnerable functions in the test set to all training samples. We normalize the formatting characters ... identify the exact copy if two strings are identical after the normalization."
  - Values:

| Benchmark | Copy (%) |
|---|---|
| BigVul | 12.7 |
| CVEFixes | 18.9 |
| CodeXGLUE | 0.6 |
| DiverseVul (hash-deduplicated) | 3.3 |
| PrimeVul | 0.0 |

  - Text: "existing benchmarks suffer from significant exact copy, up to 18.9% of samples being duplicated"; "with hash-based deduplication, DiverseVul still has 3.3% copies. This is mainly because they did not normalize formatting characters".
- **Splits behind the copy rates** (VERIFIED):
  - CodeXGLUE: its original split.
  - BigVul: "the public split [29]".
  - CVEFixes and DiverseVul: random 80/10/10.
  - [arXiv v2](https://arxiv.org/pdf/2403.18624)
- **PrimeVul size** (VERIFIED):
  - "6,968 vulnerable and 228,800 benign functions across 755 projects and 6,827 commits", covering 140 CWEs, with "5,480 such pairs".
  - Comparison with SVEN: "SVEN has only 417 vulnerable functions in C/C++, and 386 vulnerable functions in Python."
  - [arXiv v2](https://arxiv.org/pdf/2403.18624)

### Inferences
- PrimeVul's "Copy (%)" is a **cross-split leakage rate for vulnerable test functions**. It is not a dataset-wide duplicate rate like Croft's uniqueness. That is why CodeXGLUE/Devign shows 0.6% here but 10.1% non-unique (type-3 clones) in Croft. The numbers are not interchangeable.
- The 24% (CodeXGLUE) rests on 50 functions. Its breakdown, 58% irrelevant commits, points mainly at the commit-selection step (keyword filter plus annotation) in Devign. Top Score reaches the same conclusion (Q8).

### Gaps
- I did not read the ICSE 2025 IEEE Xplore version. The arXiv supplementary may not be in the camera-ready.
- The SVEN 50-sample audit does not say how many of the 50 functions were Python and how many were C/C++ (see Q9).

## Q5. ReposVul (Wang et al., ICSE Companion 2024): tangled patches, outdated patches, duplicated functions

### Takeaway
ReposVul reports **label accuracy of its own untangling method** on 50 CVE cases per language: 85% C, 90% C++, 85% Java, 80% Python. It also reports **outdated-patch ratios over time**: 20.07% of patches in 2014 down to 10.42% in 2022. In the arXiv v2 text it gives **no overall percentage of tangled patches and no duplicated-function percentage**.

### Cited Findings
- **Bibliographic record** (VERIFIED, Crossref + arXiv PDF): Xinchen Wang, Ruida Hu, Cuiyun Gao, Xin-Cheng Wen, Yujia Chen, Qing Liao. "ReposVul: A Repository-Level High-Quality Vulnerability Dataset." *Proceedings of the 2024 IEEE/ACM 46th International Conference on Software Engineering: Companion Proceedings (ICSE-Companion '24)*, Lisbon, pp. 472-483. DOI 10.1145/3639478.3647634. arXiv:2401.13169 (v2, 8 Feb 2024 read). URL read: https://arxiv.org/pdf/2401.13169 - [Crossref](https://api.crossref.org/works?query.bibliographic=ReposVul)
- **Size** (VERIFIED):
  - "6,134 CVE entries representing 236 CWE types across 1,491 projects".
  - "14,706 files from 6,897 patches".
  - Functions: "212,790 functions in C, 20,302 in C++, 2,816 in Java, and 26,308 in Python".
  - [arXiv](https://arxiv.org/pdf/2401.13169)
- **Label accuracy (RQ2, Table 5)** (VERIFIED):
  - Sample and raters: "we randomly select 50 CVE cases for each programming language. We recruit three academic researchers ... participants reach agreements on 96% for the cases".
  - Result: "ReposVul shows the labeling accuracy at 85%, 90%, 85%, and 80% on C, C++, Java, and Python, respectively."
  - Table 5 baselines: LLMs 70% / 80% / 78% / 72%, static analysis tools 82% / 84% / 78% / 74% (C / C++ / Java / Python).
  - It re-cites Croft: "20-71% of vulnerability labels are inaccurate".
  - [arXiv](https://arxiv.org/pdf/2401.13169)
- **Outdated patches (RQ4)** (VERIFIED):
  - Totals: "the overall number of patches ... ascending from 264 in 2014 to 1,132 in 2022. Concurrently, the count of outdated patches ... progressing from 53 in 2014 to 118 in 2022".
  - Ratio: "the ratio of outdated patches to total patches demonstrates a notable decrease, plummeting from 20.07% in 2014 to 10.42% in 2022."
  - By project: "The proportion of outdated patches is highest in the Linux project, attaining a rate of 16.05%", then "ImageMagick and Vim, exhibiting proportions of 10.26% and 6.58%".
  - By language: "Among these languages, C constitutes the predominant share at 70.7%, followed by Java at 14.6%, Python at 8.1%, and C++ at 6.6%". These are shares of the outdated patches, not rates.
  - [arXiv](https://arxiv.org/pdf/2401.13169)
- **Tangled patches** (VERIFIED as qualitative only): the paper defines tangled patches ("Vulnerability patches may contain vulnerability-fixing unrelated code changes") and says it uses an LLM plus static-analysis "vulnerability untangling module". No prevalence percentage was found in the text. [arXiv](https://arxiv.org/pdf/2401.13169)

### Inferences
- ReposVul's accuracy is judged per **file-level code change relevance** ("Yes"/"No" per file in the patch), not per function as in DiverseVul and PrimeVul. The 80-90% figures are therefore not directly comparable with the function-level 24-60% figures above.

### Gaps
- The arXiv v2 text gives no share of tangled patches and no share of duplicated functions. If the camera-ready (ACM DL) reports them, I did not see it.

## Q6. CleanVul (Li et al., arXiv:2411.17274): noise in vulnerability-fixing-commit data and its own correctness

### Takeaway
On a manually labelled sample of **487** function-level changes, the uncleaned GitHub VFC data had **28.7%** Correctness, meaning 28.7% of changes were genuine vulnerability fixes. VulSifter raised this to **90.6%** at threshold 3 and **97.3%** at threshold 4. The paper went through 7 arXiv versions, changed its title, and dataset sizes changed between versions. No peer-reviewed venue was confirmed; v7 still carries an ACM template placeholder.

### Cited Findings
- **Bibliographic record** (VERIFIED from the arXiv PDFs):
  - Authors: Yikun Li, Ting Zhang, Ratnadira Widyasari, Yan Naing Tun, Huu Hung Nguyen, Tan Bui, Ivana Clairine Irsan, Yiran Cheng, Xiang Lan, Han Wei Ang, Frank Liauw, Martin Weyssow, Hong Jin Kang, Eng Lieh Ouh, Lwin Khin Shar, David Lo.
  - arXiv:2411.17274.
  - Titles: v1 (26 Nov 2024) through v4 (13 Mar 2025) are titled "CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics". v7 (11 Sep 2025) is titled "CleanVul: Toward High-Quality Function-Level Vulnerability Datasets via LLM-Based Noise Reduction".
  - Venue: v7 still shows the placeholder "J. ACM 37, 4, Article 111 (August 2018)" and DOI "XXXXXXX", so no venue is established.
  - URLs read: https://arxiv.org/pdf/2411.17274 (v7), https://arxiv.org/pdf/2411.17274v1, v2, v3, v4.
  - [arXiv v7](https://arxiv.org/pdf/2411.17274); [arXiv v1](https://arxiv.org/pdf/2411.17274v1)
- **Correctness** (VERIFIED, v7 Sec. 6.1 RQ1 / Table 3):
  - Definition: "Correctness is defined as the percentage of genuine vulnerable functions in the vulnerability dataset".
  - Sample: "we randomly select a sample of 487 function-level code changes from the collected GitHub VFC dataset and manually analyze them ... a confidence level of 95% with a margin of error of ±4.4%".
  - Result: "the Correctness ... improves from 28.7% to a range of 37.5% to 97.3% on the test sample"; "At a threshold of 3, VulSifter achieves a Correctness rate of 90.6%, comparable to SVEN [10] (94.0%) and PrimeVul [5] (86.0%)."
  - The same 28.7% and 90.6% appear in v1.
  - [arXiv v7](https://arxiv.org/pdf/2411.17274)
- **Nature of the noise** (VERIFIED): "the predominant non-vulnerability changes were test-related (41.2%) and bug fixes (38.2%)"; "approximately 80% of non-vulnerability changes consist of test-related modifications and general bug fixes". [arXiv v7](https://arxiv.org/pdf/2411.17274)
- **Claim about other datasets** (VERIFIED as quoted; it is their paraphrase of PrimeVul): "According to recent work [5], most vulnerability datasets include 40% to 75% of noisy data." [arXiv v7](https://arxiv.org/pdf/2411.17274)
- **Source data** (VERIFIED, v7 Sec. 4):
  - "a comprehensive crawl of 127,063 repositories, resulting in the acquisition of 5,352,105 commits. Using the keyword-based approach proposed by Bui et al. [2], we identified 43,029 function changes".
  - Languages processed: "26,423 Java, 6,591 Python, 5,578 C, 4,000 JavaScript, 312 C#, and 125 C++ changes".
  - [arXiv v7](https://arxiv.org/pdf/2411.17274)
- **Version differences** (VERIFIED):
  - Function count: v1 "CleanVul comprises 11,632 functions"; v7 "CleanVul comprises 8,198 functions".
  - Threshold 4: v1 "8,337 vulnerability-fixing changes"; v7 "6,368 vulnerability-fixing changes".
  - Correctness figures (28.7%, 90.6%, 97.3%) are unchanged.
  - [arXiv v1](https://arxiv.org/pdf/2411.17274v1); [arXiv v7](https://arxiv.org/pdf/2411.17274)
- **Venue search** (PARTIAL): a web search found only the arXiv, GitHub and Google Scholar entries, and no acceptance notice. [GitHub yikun-li/CleanVul](https://github.com/yikun-li/CleanVul)

### Inferences
- The 28.7% baseline describes CleanVul's **own keyword-mined GitHub VFC pool**, not BigVul, Devign or other benchmarks. It is a noise estimate for keyword-mined commits in general.
- The 90.6% is pooled over languages; no per-language correctness is reported. Java dominates the processed changes (26,423 of 43,029).

### Gaps
- There is no per-language Correctness, so JavaScript and Python cannot be singled out.
- The peer-reviewed venue is unknown as of v7 (Sep 2025).

## Q7. How the standard datasets were labelled: Devign, D2A, Juliet/SARD, MegaVul, ReVeal (and their duplicates)

### Takeaway
- **Devign** labels whole commits: keyword filter, then 4 security researchers (600 man-hours, two rounds). Every function modified by a vulnerability-fix commit is labelled vulnerable.
- **D2A** labels come from Infer differential analysis. Its own 57-example check found 53% accuracy (35% without the auto-labeler), and raw Infer on OpenSSL had 7.8% true positives.
- **Juliet** is synthetic, so Croft found labels 100% correct but 83.7% non-unique.
- **MegaVul** applies deduplication and consistency filters but reports no manual accuracy audit.
- **ReVeal** measured up to 68.63% duplicates after preprocessing (Juliet), against 0.2-0.6% for raw real-world data.

### Cited Findings
- **Devign bibliographic record** (VERIFIED):
  - Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, Yang Liu. "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks." *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, Curran Associates.
  - The official NeurIPS BibTeX has an empty `pages` field.
  - arXiv:1909.03496.
  - URLs read: https://arxiv.org/pdf/1909.03496 and the NeurIPS BibTeX - [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html)
- **Devign labelling** (VERIFIED, Sec. 3.1):
  - Keyword filter: "we exclude the security-unrelated commits whose messages are not matched by a set of security-related keywords such as DoS and injection."
  - Manual labelling: "A team of four professional security researchers spent totally 600 man-hours to perform a two round data labelling and cross-verification. Given a VFC or non-CFC, based on the modified functions, we extract the source code of these functions before the commit is applied, and assign the labels accordingly."
  - Table 1 totals: 48,687 security-related commits, 23,355 VFCs, 25,332 non-VFCs, 58,965 graphs (27,652 vulnerable) over Linux, QEMU, Wireshark and FFmpeg.
  - [Devign arXiv](https://arxiv.org/pdf/1909.03496)
- **Devign as used by others** (VERIFIED): Croft's Table II lists Devign at 27,318 functions, 45.61% vulnerable. PrimeVul calls it "CodeXGLUE [8] (a.k.a. Devign [19] dataset)". [Croft](https://arxiv.org/pdf/2301.05456); [PrimeVul](https://arxiv.org/pdf/2403.18624)
- **D2A bibliographic record** (VERIFIED, Crossref): Yunhui Zheng, Saurabh Pujar, Burn Lewis, Luca Buratti, Edward Epstein, Bo Yang, Jim Laredo, Alessandro Morari, Zhong Su. "D2A: A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis." *2021 IEEE/ACM 43rd International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)*, pp. 111-120, 2021. DOI 10.1109/ICSE-SEIP52600.2021.00020. arXiv:2102.07995. URL read: https://arxiv.org/pdf/2102.07995 - [Crossref](https://api.crossref.org/works/10.1109/ICSE-SEIP52600.2021.00020)
- **D2A labels and precision** (VERIFIED):
  - Scale: "Out of 349,373,753 issues reported by the static analyzer, after deduplication, we labeled 18,653 unique issues as positives and 1,276,970 unique issues as negatives."
  - Abstract-level claim: "we randomly selected and manually reviewed 57 examples. The result shows that D2A improves the label accuracy from 7.8% to 53%."
  - Detailed: "57 examples (41 positives, 16 negatives) with a focus on positives ... Each example was independently reviewed by 2 reviewers ... On this biased sample set, the accuracy with and without the auto-labeler is 53% and 35% respectively ... Without auto-labeler, the accuracy was only 7.8% on the set of 166 security-related examples" (OpenSSL study: "13 (7.8%) issues are true positives and 92.2% are false positives").
  - [D2A arXiv](https://arxiv.org/pdf/2102.07995)
- **Juliet/SARD** (VERIFIED):
  - Croft Table III: accuracy 1.000, uniqueness 0.163, consistency 0.750, completeness 1.000.
  - PrimeVul: "synthetic datasets do not adequately capture the complex and nuanced nature of vulnerabilities in real-world code".
  - [Croft](https://arxiv.org/pdf/2301.05456); [PrimeVul](https://arxiv.org/pdf/2403.18624)
- **MegaVul bibliographic record** (VERIFIED, Crossref + arXiv PDF): Chao Ni, Liyu Shen, Xiaohu Yang, Yan Zhu, Shaohua Wang. "MegaVul: A C/C++ Vulnerability Dataset with Comprehensive Code Representations." *Proceedings of the 21st International Conference on Mining Software Repositories (MSR '24)*, Lisbon, pp. 738-742. DOI 10.1145/3643991.3644886. arXiv:2406.12415. The arXiv title block lists only the first four authors, and Crossref lists five. URL read: https://arxiv.org/pdf/2406.12415 - [Crossref](https://api.crossref.org/works/10.1145/3643991.3644886)
- **MegaVul quality handling** (VERIFIED): "Deduplication Filters aim to filter out duplicated functions"; there are also anomaly, test-related and "Other Filters" ("to ensure label consistency and prevent the same function from being annotated as both vulnerable and non-vulnerable"). The paper concedes that residual duplication remains: "it is challenging to automatically determine whether their content is identical, resulting in duplication in the dataset." No manual label-accuracy figure was found. [MegaVul arXiv](https://arxiv.org/pdf/2406.12415)
- **ReVeal duplicates** (VERIFIED, Table 4 "Percentage of duplicate samples in datasets"):
  - Juliet (Russell et al. preprocessing): 68.63%.
  - NVD+SARD: VulDeePecker 67.33%, SySeVR 61.99%.
  - Draper (Russell et al.): 6.07 / 2.99.
  - ReVeal dataset: VulDeePecker 25.85%, SySeVR 25.56%, Russell et al. 8.93%.
  - FFMPeg+Qemu: VulDeePecker 19.58%, SySeVR 22.10%, Russell et al. 20.54%.
  - Raw data: "REVEAL dataset had only 0.6%, and FFMPeg+Qemu had 0.2%".
  - Summary: "The training and testing data in most existing approaches contain duplicates (up to 68%)".
  - [ReVeal arXiv](https://arxiv.org/pdf/2009.07235)

### Inferences
- The "manual labelling" in Devign validates **commits**, not individual functions. Every function touched by an accepted commit inherits the vulnerable label, which is the mechanism the later audits (PrimeVul 24%, Top Score 47-50%) blame.
- Duplication figures depend heavily on the definition, so each must be quoted with its definition. For Devign/FFmpeg+QEMU:
  - 0.2% raw exact duplicates (ReVeal).
  - 0.6% test-to-train exact copies (PrimeVul).
  - 10.1% type-3 clones (Croft).
  - 19.6-22.1% after slicing and tokenisation (ReVeal).

### Gaps
- The Devign NeurIPS page numbers are not in the official NeurIPS BibTeX (empty field). I did not verify the page range some citation managers give.
- I found no manual label-accuracy audit of MegaVul.

## Q8. Related data-quality critiques with numbers

### Takeaway
- **Top Score** (ISSTA 2025) independently finds only 39-65% of vulnerable labels correct in BigVul, Devign and DiverseVul. It finds **0** of the truly vulnerable functions decidable without context.
- **Arp et al.** report label inaccuracy fully present in 10% of 30 top-tier security ML papers.
- **Jimenez et al.** show MCC collapsing under realistic labelling.
- **Croft et al. (MSR'22)** find over 3x more latent than reported vulnerabilities.
- **Real-Vul** reports precision drops of up to 95 points on realistic data.

### Cited Findings
- **Top Score on the Wrong Exam** (VERIFIED):
  - Bibliographic record (Crossref + arXiv PDF): Niklas Risse, Jing Liu, Marcel Böhme. "Top Score on the Wrong Exam: On Benchmarking in Machine Learning for Vulnerability Detection." *Proceedings of the ACM on Software Engineering*, Vol. 2, Issue ISSTA, pp. 388-410, 2025. DOI 10.1145/3728887. arXiv:2408.12986: v1 (23 Aug 2024; authors Risse and Böhme only), v2 (23 Apr 2025; adds Jing Liu). URLs read: https://arxiv.org/pdf/2408.12986 and https://arxiv.org/pdf/2408.12986v1 - [Crossref](https://api.crossref.org/works/10.1145/3728887)
  - Method (v2): "From each of the three datasets, we randomly selected 100 samples labeled as vulnerable"; "two Software Security researchers (co-authors of this paper) reviewed each of the 300 functions independently", taking "more than 150 hours". [arXiv v2](https://arxiv.org/pdf/2408.12986)
  - Result (v2): "Out of the 100 functions per dataset that were originally labeled as vulnerable, only 39%-65% (Devign: 47%, BigVul: 39%, DiverseVul: 65%) actually contain security vulnerabilities." [arXiv v2](https://arxiv.org/pdf/2408.12986)
  - Result (v1, differs): "only 38%-64% (Devign: 50%, BigVul: 38%, DiverseVul: 64%)". [arXiv v1](https://arxiv.org/pdf/2408.12986v1)
  - Context dependence (v2, Fig. 6): every truly vulnerable function was "Context-dependent" (BigVul 39/39, Devign 47/47, DiverseVul 65/65; "Context-independent: 0"). Abstract: the vulnerability of a function "cannot be decided without further context for more than 90% of functions". [arXiv v2](https://arxiv.org/pdf/2408.12986)
  - Reasons (v2, Table I "Reasons for Inaccurate Labels"):
    - BigVul: patch-commit identification 13 (21%), structural changes 29 (48%), unrelated changes 19 (31%).
    - Devign: 27 (54%) / 19 (38%) / 4 (8%).
    - DiverseVul: 1 (3%) / 23 (66%) / 11 (31%).
    - Text: "We observe that 54% of falsely labeled functions in our sample of the Devign dataset originate from this automatic identification process" (keyword filtering).
    - [arXiv v2](https://arxiv.org/pdf/2408.12986)
  - On Croft: "they found at least 20% of labels for the Devign dataset and 45.7% of labels for the BigVul dataset to be inaccurate". The gap is attributed to Croft counting callers as vulnerable. [arXiv v2](https://arxiv.org/pdf/2408.12986)
  - Dataset popularity: "BigVul [29] and Devign [30] were both used by 36 papers, and ReVeal [31] by 22 papers." [arXiv v2](https://arxiv.org/pdf/2408.12986)
- **Arp et al., "Dos and Don'ts of Machine Learning in Computer Security"** (VERIFIED):
  - Bibliographic record: Daniel Arp, Erwin Quiring, Feargus Pendlebury, Alexander Warnecke, Fabio Pierazzi, Christian Wressnegger, Lorenzo Cavallaro, Konrad Rieck. *31st USENIX Security Symposium (USENIX Security 22)*, Boston, Aug 2022, pp. 3971-3988, ISBN 978-1-939133-31-1. arXiv:2010.09470. URLs read: https://arxiv.org/pdf/2010.09470 and the USENIX page - [USENIX](https://www.usenix.org/conference/usenixsecurity22/presentation/arp)
  - Pitfall box, verbatim: "P2 - Label Inaccuracy. The ground-truth labels required for classification tasks are inaccurate, unstable, or erroneous ... 10% present". Prevalence is measured over "30 papers from top-tier security conferences within the past 10 years".
  - Wider results: "The most prevalent pitfalls are sampling bias (P1) and data snooping (P3), which are at least partly present in 90 % and 73 % of the papers".
  - [arXiv](https://arxiv.org/pdf/2010.09470)
- **Jimenez et al. FSE 2019** (PARTIAL; abstract via Semantic Scholar API, metadata via Crossref):
  - Bibliographic record: Matthieu Jimenez, Renaud Rwemalika, Mike Papadakis, Federica Sarro, Yves Le Traon, Mark Harman. "The importance of accounting for real-world labelling when predicting software vulnerabilities." *Proc. 27th ACM ESEC/FSE 2019*, pp. 695-705. DOI 10.1145/3338906.3338941.
  - Abstract: "1,898 real-world vulnerabilities reported in 74 releases of three security-critical open source systems"; "MCC mean values of predictive effectiveness drop from 0.77, 0.65 and 0.43 to 0.08, 0.22, 0.10 for Linux Kernel, OpenSSL and Wiresark, respectively."
  - [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3338906.3338941?fields=title,abstract); [Crossref](https://api.crossref.org/works/10.1145/3338906.3338941); replication package: [GitHub kabinja/fse2019](https://github.com/kabinja/fse2019)
- **Croft, Babar, Chen, "Noisy Label Learning for Security Defects"** (VERIFIED):
  - Bibliographic record: *Proc. 19th International Conference on Mining Software Repositories (MSR '22)*, Pittsburgh, pp. 435-447. DOI 10.1145/3524842.3528446. arXiv:2203.04468. URL read: https://arxiv.org/pdf/2203.04468 - [Crossref](https://api.crossref.org/works/10.1145/3524842.3528446)
  - Latent vulnerabilities, on Mozilla Firefox file-level labels: "despite this incomplete knowledge, we observed over three times as many latent vulnerabilities than the available reported vulnerabilities. Hence, label noise is a significant factor in SV datasets."
  - Silent fixes, cited from Jiang et al.: "over 65% of vulnerability patches were silently reported".
  - The proposed method "improves AUC and recall of baselines by up to 8.9% and 23.4%".
  - [arXiv](https://arxiv.org/pdf/2203.04468)
- **Risse & Böhme, "Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection"** (VERIFIED bibliographic record; content checked by keyword search only):
  - Bibliographic record: *33rd USENIX Security Symposium (USENIX Security 24)*, Philadelphia, Aug 2024, pp. 4247-4264, ISBN 978-1-939133-44-1. arXiv:2306.17193.
  - Content: the paper is about semantic-preserving and label-inverting transformations, and it introduced VulnPatchPairs. My keyword search found no label-accuracy audit in it.
  - PrimeVul later measured VulnPatchPairs at **36%** label accuracy (VERIFIED).
  - [USENIX](https://www.usenix.org/conference/usenixsecurity24/presentation/risse); [arXiv](https://arxiv.org/pdf/2306.17193); [PrimeVul](https://arxiv.org/pdf/2403.18624)
- **ReVeal** (VERIFIED):
  - Bibliographic record: Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, Baishakhi Ray. "Deep Learning Based Vulnerability Detection: Are We There Yet?" *IEEE Transactions on Software Engineering* 48(9):3280-3296, 2022. DOI 10.1109/TSE.2021.3087402. arXiv:2009.07235.
  - Duplication numbers are given in Q7.
  - [Crossref](https://api.crossref.org/works/10.1109/TSE.2021.3087402)
- **Real-Vul** (VERIFIED):
  - Bibliographic record: Partha Chakraborty, Krishna Kanth Arumugam, Mahmoud Alfadel, Meiyappan Nagappan, Shane McIntosh. "Revisiting the Performance of Deep Learning-Based Vulnerability Detection on Realistic Datasets." *IEEE Transactions on Software Engineering* 50(8):2163-2177, 2024. DOI 10.1109/TSE.2024.3423712. arXiv:2407.03093.
  - Abstract: "precision declining by up to 95 percentage points and F1 scores dropping by up to 91 percentage points".
  - Threats to validity: "It is possible that some vulnerable samples in the BigVul dataset are mislabelled. However, the labeled samples in the dataset were manually verified by Fan et al. [7]".
  - [arXiv](https://arxiv.org/pdf/2407.03093); [Crossref](https://api.crossref.org/works?query.bibliographic=Revisiting+the+Performance+of+Deep+Learning-Based+Vulnerability+Detection+on+Realistic+Datasets)

### Inferences
- Real-Vul's claim that BigVul was "manually verified by Fan et al." conflicts with three independent audits: 54.3% (Croft), 25% (DiverseVul) and 38-39% (Top Score). Treat it as unsupported. I did not check what the original Big-Vul (MSR 2020) paper itself claims.
- Across the audits that re-examine the same datasets, the ranking holds even though the levels differ: BigVul is lowest, DiverseVul and CVEFixes are higher, and SVEN and PrimeVul are highest.

### Gaps
- Arp et al.'s Figure 3 also shows a "partly present" share for P2 that I could not extract reliably from the PDF text. Only the "10% present" header value is verified.
- Jimenez et al.: I did not read the full text (ACM DL returned 403).
- The original Big-Vul paper (Fan et al., MSR 2020) was not read, so its own statements about manual verification remain unchecked.

## Q9. Python-specific: is SVEN's Python subset manually verified, and are there label-quality measurements for Python VD data?

### Takeaway
SVEN (CCS 2023) was built by **manual inspection** of commits from Big-Vul, CrossVul and VUDENC. It has 1,606 programs (803 pairs): **760 Python** and 846 C/C++. PrimeVul later audited 50 SVEN vulnerable functions and found **94%** correct, but it does not say how many were Python. The only per-language Python label-accuracy figure I found is ReposVul's **80%** on 50 Python CVE cases. No Python-only audit of SVEN exists in the sources read.

### Cited Findings
- **SVEN bibliographic record** (VERIFIED, Crossref + arXiv PDF): Jingxuan He, Martin Vechev. "Large Language Models for Code: Security Hardening and Adversarial Testing." *Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (CCS '23)*, Copenhagen, Nov 26-30 2023, pp. 1865-1879. DOI 10.1145/3576915.3623175. arXiv:2302.05319. URL read: https://arxiv.org/pdf/2302.05319 - [Crossref](https://api.crossref.org/works/10.1145/3576915.3623175)
- **SVEN curation** (VERIFIED, Sec. 4.3):
  - "we include CrossVul [58] and Big-Vul [34] ... We also include VUDENC [76] because it focuses on Python while the majority of programs in CrossVul and Big-Vul are in C/C++."
  - "VUDENC [76] applies keyword-matching on commit messages to collect its dataset, which produces many false positives."
  - "To improve data quality, we perform manual inspection on the commits of [34, 58, 76] for our target CWEs. Among those commits, our inspection extracts code pairs that are true security fixes and excludes quality issues discussed above."
  - "It consists of 1,606 programs (i.e., 803 pairs). Each program is a function written in C/C++ or Python."
  - [arXiv](https://arxiv.org/pdf/2302.05319)
- **SVEN Table 1** (VERIFIED):
  - Overall: "overall 1606, py: 760, c/c++: 846, train: 1440, val: 166".
  - CWEs with Python programs: CWE-089 408 (py: 408); CWE-078 212 (py: 204, c/c++: 8); CWE-022 114 (py: 66, c/c++: 48); CWE-079 100 (py: 82, c/c++: 18).
  - [arXiv](https://arxiv.org/pdf/2302.05319)
- **PrimeVul's audit of SVEN** (VERIFIED): SVEN is 94% correct, 6% irrelevant, 0% spread and 0% relevant-consistency (Suppl. Table IX), from "50 vulnerable functions randomly sampled from each dataset, including ... SVEN". Also: "SVEN has only 417 vulnerable functions in C/C++, and 386 vulnerable functions in Python." [PrimeVul](https://arxiv.org/pdf/2403.18624)
- **Python per-language accuracy elsewhere** (VERIFIED): ReposVul reports "80% ... on ... Python" from 50 Python CVE cases (Table 5), with Python static tools at 74% and LLMs at 72%. [ReposVul](https://arxiv.org/pdf/2401.13169)
- **CleanVul Python share** (VERIFIED): 6,591 Python changes were processed, but Correctness is reported only pooled across languages. [CleanVul v7](https://arxiv.org/pdf/2411.17274)
- **Local project data** (VERIFIED from repo file): the project's default target set `data/sven_python_folds_norm` has "760 (380/380)" rows. This equals SVEN's Table 1 Python count ("py: 760"). Source: /drive1/cuongtm/ntat/MultiVD/CLAUDE.md §6.

### Inferences
- SVEN's Python subset was manually inspected at construction. That is the only verification step documented for it.
  - The SVEN paper, in the sections I read, reports no inter-rater agreement and no error-rate estimate for that curation.
  - PrimeVul's 94% is the only external audit, it is pooled across C/C++ and Python, and it rests on 50 functions.
  - So "manually curated, ~94% correct (PrimeVul, n=50, language mix unstated)" is the strongest defensible statement for the 760-function Python target set.
- SVEN's Python CWEs are concentrated in CWE-089 (408 of 760), so SQL injection makes up about 54% of the Python programs (my arithmetic from Table 1).

### Gaps
- There is no published Python-only label audit of SVEN. PrimeVul's released label-analysis artifact (promised in the paper) might show which of its 50 SVEN samples were Python, but I did not retrieve it.
- There is no published duplication or near-duplicate measurement specific to SVEN's Python subset. The project's own note (CLAUDE.md §6) says about 40% of test rows in the random split have a near-duplicate in train. That is a local measurement, not a published one.
