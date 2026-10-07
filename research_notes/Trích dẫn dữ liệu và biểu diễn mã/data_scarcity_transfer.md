# Labeled vulnerability-data scarcity (esp. Python) and transfer as the answer: verified citations

Status legend used on every item:
- **VERIFIED** = full text (or the exact passage) was read at the URL given; quotes are verbatim.
- **PARTIAL** = only abstract / metadata read (publisher page paywalled); quotes are from the abstract.
- **UNVERIFIED** = seen only in a search snippet or secondary source; do not quote without re-checking.
- Bibliographic fields (authors, venue, volume, pages, DOI) were cross-checked against Crossref (`https://api.crossref.org/works/<DOI>`) and/or Semantic Scholar (`https://api.semanticscholar.org/graph/v1/paper/DOI:<DOI>`) unless noted.
- "Computed" marks a ratio I derived from the paper's own counts; the paper does not state it.
- Preprints are flagged **[PREPRINT]**. Venues of doubtful quality are flagged **[VENUE?]**.

Context of the citing project (for the report writer): Phase 1 pre-trains on C/C++ (PrimeVul) and Java/JavaScript (CleanVul). Phase 2 fine-tunes on the 760 Python functions of SVEN.

---

## Q1. Survey-level and empirical statements that labeled real-world VD data is scarce and highly imbalanced (claim a)

### Takeaway
Claim (a) is well supported, but mostly by **empirical papers and recent mappings**, not by the two classic surveys the brief named. For Lin et al. (Proc. IEEE 2020) and Ghaffarian & Shahriari (CSUR 2017) I could verify only metadata and abstracts, and neither abstract states scarcity in citable words. The strongest verbatim quotes and numbers come from ReVeal (TSE 2022), Lin et al. TII 2018, Nguyen et al. PAKDD 2020 / TOSEM 2024, Le & Babar ESEM 2024 and the 2026 label-efficiency mapping. Real-world vulnerable shares range from about 0.5% to 9%.

### Cited Findings

**Chakraborty, Krishna, Ding, Ray. "Deep Learning Based Vulnerability Detection: Are We There Yet?" IEEE Trans. Software Eng. 48(9):3280-3296, Sept 2022 (early access 2021). DOI 10.1109/TSE.2021.3087402. arXiv:2009.07235. VERIFIED** (arXiv full text).
- Quote: "Data Imbalance. Existing approaches do not alleviate the class imbalance problem [19], [20] of real-world vulnerability distribution as non-vulnerable code is much more frequent than the vulnerable ones." - [arXiv PDF](https://arxiv.org/pdf/2009.07235)
- Quote: "the proportion of vulnerable examples in comparison to the non-vulnerable one in real world dataset is extremely low [8]. When a model is trained on such imbalanced dataset, models tend to be biased by the non-vulnerable examples." - [arXiv PDF](https://arxiv.org/pdf/2009.07235)
- Quote on Devign's FFmpeg+Qemu: "the ratio of vulnerable and non-vulnerable examples is approximately 45%-55%, which does not reflect the real world distribution of vulnerable code." - [arXiv PDF](https://arxiv.org/pdf/2009.07235)
- Table 1 numbers: ReVeal dataset (Chromium + Debian) has **18,169 functions, 9.16% vulnerable**. Draper has 1,274,366 functions, 6.46% vulnerable. FFMPeg+Qemu has 22,361 functions, 45.02% vulnerable. - [arXiv PDF](https://arxiv.org/pdf/2009.07235)
- Quote: "The training and testing data in most existing approaches contain duplicates (up to 68%)". Abstract: performance "drops by more than 50%" in a realistic setting. - [arXiv PDF](https://arxiv.org/pdf/2009.07235)
- Note: the arXiv copy's header reads "IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. TBD, 2020". Cite the published version: TSE 48(9), 2022 ([Crossref](https://api.crossref.org/works/10.1109/TSE.2021.3087402)).

**Lin, Zhang, Luo, Pan, Xiang, De Vel, Montague. "Cross-Project Transfer Representation Learning for Vulnerable Function Discovery." IEEE Trans. Industrial Informatics 14(7):3289-3297, July 2018. DOI 10.1109/TII.2018.2821768. PARTIAL** (abstract).
- Quote: "its potential is often severely compromised at the early stage of a software project when we face a shortage of high-quality training data". - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TII.2018.2821768)
- Quote: "we manually labeled 457 vulnerable functions and collected 30 000+ nonvulnerable functions from six open-source projects". Computed: at most about 1.5% vulnerable. - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TII.2018.2821768)

**Nguyen, Le, De Vel, Montague, Grundy, Phung. "Dual-Component Deep Domain Adaptation: A New Approach for Cross Project Software Vulnerability Detection." PAKDD 2020, LNCS, pp. 699-711. DOI 10.1007/978-3-030-47426-3_54 (PMC7206170). VERIFIED** (open full text on PMC).
- Abstract quote: "One of the most crucial issues in SVD is coping with the scarcity of labeled vulnerabilities in projects that require the laborious manual labeling of code by software security experts." - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206170/)
- Body quote: "Labelled vulnerable code is needed to train these models, and the process of labeling vulnerable source code is very tedious, time-consuming, error-prone, and challenging even for domain experts. This has led to few labeled projects compared with the vast volume of unlabeled ones." - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206170/)
- Per-project counts (vulnerable / non-vulnerable functions): FFmpeg 187 / 5,427; LibTIFF 81 / 695; LibPNG 43 / 551; **VLC 25 / 5,548**; **Pidgin 42 / 8,268**. Computed vulnerable shares: VLC 0.45%, Pidgin 0.51%. - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206170/)

**Nguyen, Le, Tantithamthavorn, Grundy, Phung. "Deep Domain Adaptation With Max-Margin Principle for Cross-Project Imbalanced Software Vulnerability Detection." ACM TOSEM 33(6):1-34, 2024. DOI 10.1145/3664602. PARTIAL** (abstract).
- Quote: "(ii) tackling the scarcity of labeled vulnerability datasets that conventionally need laborious labeling effort by experts." Transfer is "from imbalanced labeled into imbalanced unlabeled projects". F1 improves "from 1.83% to 6.25% compared to the second highest method". - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3664602)

**Le, Babar. "Automatic Data Labeling for Software Vulnerability Prediction Models: How Far Are We?" ESEM 2024 (accepted per arXiv comment). arXiv:2407.17803. PARTIAL** (abstract; DOI not checked).
- Quote: "Software Vulnerability (SV) prediction needs large-sized and high-quality data to perform well. Current SV datasets mostly require expensive labeling efforts by experts (human-labeled) and thus are limited in size." - [arXiv abs](https://arxiv.org/abs/2407.17803)
- Numbers: "50+% of the auto-labeled SVs are noisy". Yet models using them "can perform up to 22% and 90% better in Matthews Correlation Coefficient and Recall". - [arXiv abs](https://arxiv.org/abs/2407.17803)

**Khalal, Fettal, Labiod, Nadif. "When Labels Are Scarce: A Systematic Mapping of Label-Efficient Code Vulnerability Detection." arXiv:2604.00079, 31 Mar 2026. [PREPRINT]. VERIFIED** (full text).
- Abstract quote: "dependable vulnerability labeling remains expensive, noisy, and uneven across projects, languages, and CWE types". - [arXiv PDF](https://arxiv.org/pdf/2604.00079)
- Body quote: "High-quality vulnerability labels are expensive to obtain, unevenly distributed across CWE types, and prone to noise." The same text lists "evaluation pitfalls, including duplication-prone random splits and severe class imbalance". - [arXiv PDF](https://arxiv.org/pdf/2604.00079)

**Shimmi, Okhravi, Rahimi. "AI-Based Software Vulnerability Detection: A Systematic Literature Review." arXiv:2506.10280, June 2025, covering 2018-2023. [PREPRINT]. VERIFIED** (full text).
- §7.1.1 quote: "A significant challenge with current datasets is the scarcity of real-world data suitable for training purposes." - [arXiv PDF](https://arxiv.org/pdf/2506.10280)
- §7.1.2 quote: "The current datasets contain more non-vulnerable records than vulnerable ones as we observed in the primary papers used in our work in Table 3. The same observation is also mentioned by Ghaffarian and Shahriar [46]". - [arXiv PDF](https://arxiv.org/pdf/2506.10280)

**Large real-world C/C++ corpora: vulnerable shares (source data of the citing project)**
- PrimeVul: Ding, Fu, Ibrahim, Sitawarin, Chen, Alomair, Wagner, Ray, Chen. "Vulnerability Detection with Code Language Models: How Far Are We?" ICSE 2025, pp. 1729-1741. DOI 10.1109/ICSE55347.2025.00038. arXiv:2403.18624. **VERIFIED**.
  - Quote: "PrimeVul contains 6,968 vulnerable and 228,800 benign functions, covering 140 CWEs". Computed: 2.96% vulnerable.
  - Abstract: "a state-of-the-art 7B model scored 68.26% F1 on BigVul but only 3.09% F1 on PrimeVul."
  - Source: [arXiv PDF](https://arxiv.org/pdf/2403.18624)
- DiverseVul: Chen, Ding, Alowain, Chen, Wagner. RAID 2023. arXiv:2304.00409. **PARTIAL** (abstract; DOI not checked).
  - Quote: "18,945 vulnerable functions spanning 150 CWEs and 330,492 non-vulnerable functions extracted from 7,514 commits". Computed: 5.4% vulnerable.
  - Also: "increasing the volume of training data may not further improve the performance ... but might be useful to improve the generalization ability to unseen projects."
  - Source: [arXiv abs](https://arxiv.org/abs/2304.00409)

**Lin, Wen, Han, Zhang, Xiang. "Software Vulnerability Detection Using Deep Neural Networks: A Survey." Proceedings of the IEEE 108(10):1825-1848, Oct 2020. DOI 10.1109/JPROC.2020.2993293. PARTIAL** (metadata + abstract only; closed access per Semantic Scholar `isOpenAccess: false`).
- The abstract says only: "We also identify the challenges in this new field and share our views of potential research directions." It has no citable sentence on scarcity or imbalance. - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/JPROC.2020.2993293)

**Ghaffarian, Shahriari. "Software Vulnerability Analysis and Discovery Using Machine-Learning and Data-Mining Techniques: A Survey." ACM Computing Surveys 50(4):1-36, 2017. DOI 10.1145/3092566. PARTIAL** (metadata + abstract; `CLOSED` per Semantic Scholar).
- The abstract ([OpenAlex](https://api.openalex.org/works/doi:10.1145/3092566)) mentions "challenges and some uncharted territories" but not scarcity.
- Its imbalance observation is known only secondhand, via Shimmi et al. §7.1.2 above. That makes it **UNVERIFIED** as a direct quote.

### Inferences
- For a one-sentence claim (a), the safest pairing is ReVeal (TSE 2022) for imbalance, with its 9.16% figure and the "extremely low" quote, plus Nguyen et al. PAKDD 2020 for scarcity and labeling cost. Add Khalal et al. 2026 if a recent source is wanted, flagged as a preprint.
- Vulnerable shares in real-world function-level data run from about 0.45% (VLC) to 9.16% (ReVeal), all computed from or stated in the papers above. The citing project balances its Phase 1 sources to 50/50, which departs from the natural distribution and should be stated as a design choice.

### Gaps
- No verbatim scarcity or imbalance sentence could be pulled from Lin et al. 2020 or Ghaffarian & Shahriari 2017 (both paywalled). Cite them only for "surveys of the field", or get the PDFs through the library before quoting.
- I found no 2023-2026 survey that gives a **single numeric measure of scarcity**. The recent ones give study counts (Shimmi: 91 C/C++ papers, see Q2) or qualitative statements. Karim & Akter, "Vulnerability datasets for software security: A survey of existing resources, challenges, and future directions", Computers & Security 167:104926, 2026, DOI 10.1016/j.cose.2026.104926, exists per [Crossref](https://api.crossref.org/works?query.bibliographic=Vulnerability+datasets+for+software+security+A+survey), but I could not read its content (UNVERIFIED).

---

## Q2. Python VD datasets and their sizes; is Python data much smaller than C/C++ data? (claim b)

### Takeaway
All of the project's notes on SVEN check out exactly: 1,606 programs = 803 pairs, py 760 / c/c++ 846, 9 CWEs. In addition, the 760 Python programs cover only **4 of the 9 CWEs**. Claim (b) holds against the large C/C++ corpora and inside CVEfixes-derived splits: in MVD, Python has 779 vulnerable functions against 6,311 for C/C++. It does **not** hold universally. In REEF and in CleanVul's raw commit pool, Python is comparable to or larger than C. Word the claim as "curated, function-level Python VD data is much smaller than the C/C++ corpora", not "Python data is always smaller". The "Python scored lowest, 0.6626" note is correct, but it applies to one model (GPT-4o, accuracy), and the margin over JavaScript is only 0.0016.

### Cited Findings

**SVEN: He, Vechev. "Large Language Models for Code: Security Hardening and Adversarial Testing." ACM CCS 2023, pp. 1865-1879. DOI 10.1145/3576915.3623175. arXiv:2302.05319 (v5). VERIFIED** (full text).
- Quote: "It consists of 1,606 programs (i.e., 803 pairs). Each program is a function written in C/C++ or Python. We randomly split the dataset by a ratio of 9:1 into training and validation." - [arXiv PDF v5](https://arxiv.org/pdf/2302.05319v5)
- Table 1 per-CWE counts:
  - 089: py 408
  - 125: c/c++ 290
  - 078: py 204, c/c++ 8
  - 476: c/c++ 156
  - 416: c/c++ 128
  - 022: py 66, c/c++ 48
  - 787: c/c++ 112
  - 079: py 82, c/c++ 18
  - 190: c/c++ 86
  - **overall 1606, "py: 760, c/c++: 846", train 1440, val 166**
  - So the Python part covers only CWE-089/078/022/079 (408+204+66+82 = 760).
  - Source: [arXiv PDF v5](https://arxiv.org/pdf/2302.05319v5)
- Quote: "We also include VUDENC [76] because it focuses on Python while the majority of programs in CrossVul and Big-Vul are in C/C++." - [arXiv PDF v5](https://arxiv.org/pdf/2302.05319v5)
- Quote: "Our data construction relies on manual effort and deliberately excludes samples that do not meet our quality criteria, thus prioritizing quality over quantity." The baseline set had "∼19x more samples". - [arXiv PDF v5](https://arxiv.org/pdf/2302.05319v5)
- CWE selection criterion includes "we are able to extract sufficient (>40) security fixes for them". - [arXiv PDF v5](https://arxiv.org/pdf/2302.05319v5)
- Caveat: SVEN was built for **secure code generation** (prefix tuning), not as a VD benchmark.

**PyVul: Quan, Wang, Li, Zhuo, Chen, Du. "An Empirical Study of Vulnerabilities in Python Packages and Their Detection." arXiv:2509.04260, 4 Sep 2025. [PREPRINT]. VERIFIED** (full text).
- Quote: "due to a lack of specifically curated Python datasets, most existing studies have focused on other programming languages, particularly C/C++." - [arXiv PDF](https://arxiv.org/pdf/2509.04260)
- PyVul contains "1,157 commit-level and 2,082 function-level vulnerabilities". It is "comparable to the human-annotated small dataset, SVEN [33], which contains only 380 vulnerable functions." - [arXiv PDF](https://arxiv.org/pdf/2509.04260)
- Prior Python data per their Table 1: CVEfixes has 508 commits / 1,360 Python functions at 48.3% label accuracy. CrossVul has 319 commits / 777 functions at 51.0%. SVEN has 143 commits / 380 vulnerable functions at 96.3%. - [arXiv HTML](https://arxiv.org/html/2509.04260v1)
- **Conflict:** PyVul writes "SVEN ... contains 808 pairs ... 380 pairs refer specifically to Python functions". SVEN's own Table 1 says 803 pairs. Use 803 (primary source). The 380 Python pairs are consistent with 760 / 2.

**VUDENC: Wartschinski, Noller, Vogel, Kehrer, Grunske. "VUDENC: Vulnerability Detection with Deep Learning on a Natural Codebase for Python." Information and Software Technology 144:106809, April 2022. DOI 10.1016/j.infsof.2021.106809. arXiv:2201.08441. PARTIAL** (abstract).
- Quote: "we used 1,009 vulnerability-fixing commits from different GitHub repositories that contain seven different types of vulnerabilities (SQL injection, XSS, Command injection, XSRF, Remote code execution, Path disclosure, Open redirect)". Reported metrics: "recall of 78%-87%, a precision of 82%-96%, and an F1 score of 80%-90%". - [arXiv abs](https://arxiv.org/abs/2201.08441)
- Label-quality caveat from SVEN: "VUDENC [76] applies keyword-matching on commit messages to collect its dataset, which produces many false positives." - [SVEN arXiv PDF](https://arxiv.org/pdf/2302.05319v5)

**Bagheri, Hegedűs. "A Comparison of Different Source Code Representation Methods for Vulnerability Prediction in Python." QUATIC 2021, Communications in Computer and Information Science, pp. 267-281. DOI 10.1007/978-3-030-85347-1_20. arXiv:2108.02044. VERIFIED** (arXiv full text).
- Quote: "We ended up collecting approximately 70k commits yielding to 140k Python code snippets (vulnerable and fixed together) from 14k different Python projects." These are keyword-mined candidates before filtering. - [arXiv PDF](https://arxiv.org/pdf/2108.02044)
- Results: BERT+LSTM "achieved the best overall accuracy(93.8%)". The paper also says XSS recall was lower "mostly because finding good vulnerable dataset of it is difficult and we think that we didn't train it with enough data." - [arXiv PDF](https://arxiv.org/pdf/2108.02044)

**CVEfixes, Python share.** Base paper: Bhandari, Naseer, Moonen. "CVEfixes: automated collection of vulnerabilities and their fixes from open-source software." PROMISE 2021, pp. 30-39. DOI 10.1145/3475960.3475985. arXiv:2107.08760.
- Base paper, VERIFIED: "5365 CVE records for 1754 open-source projects that were addressed in a total of 5495 vulnerability fixing commits." The arXiv text I read has **no per-language function table**. - [arXiv PDF](https://arxiv.org/pdf/2107.08760)
- Per-language counts from CVEfixes, as extracted in Al Atiiq, Gehrmann, Dahlén, "Vulnerability Detection in Popular Programming Languages with Language Models", arXiv:2412.15905, Dec 2024 [PREPRINT], **VERIFIED**. Table 1, vulnerable / non-vulnerable / total:
  - C/C++: 8,299 / 11,761 / 20,060
  - **Python: 2,775 / 4,793 / 7,568**
  - Java: 3,102 / 5,285 / 8,387
  - PHP: 4,758 / 23,499 / 28,257
  - Go: 1,880 / 4,403 / 6,283
  - JavaScript: first 100,000 of 174,928 entries used
  - Computed: C/C++ has about 3.0x as many vulnerable functions as Python.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2412.15905)
- Al Atiiq et al. abstract quote: "most studies have focused on the C/C++ programming language, with limited attention given to other popular languages." Python F1 on the CVEfixes test data was 45.57%-53.27% across four models (from the HTML fetch). - [arXiv HTML](https://arxiv.org/html/2412.15905v1)

**MVD: Zhang, Le, Babar. "MVD: A Multi-Lingual Software Vulnerability Detection Framework." arXiv:2412.06166, Dec 2024. [PREPRINT] (also on Qeios). VERIFIED** (full text).
- Table I, vulnerable / non-vulnerable (% vulnerable), curated with CVEfixes tooling:
  - **Python 779 / 10,801 (6.7%)**
  - **C/C++ 6,311 / 116,725 (5.1%)**
  - Java 789 / 10,687 (6.9%)
  - C# 332 / 1,280 (20.6%)
  - JavaScript 2,969 / 28,207 (9.5%)
  - TypeScript 151 / 1,760 (7.9%)
  - Computed: C/C++ has 8.1x as many vulnerable functions as Python and 10.6x as many functions in total.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2412.06166)
- Quote: "It is evident that the number of vulnerable functions was significantly smaller than that of nonvulnerable ones, confirming our argument about the existence of class imbalance in multi-lingual vulnerability prediction." - [arXiv PDF](https://arxiv.org/pdf/2412.06166)

**Research concentration on C/C++ (study counts)**
- Shimmi et al. 2025 SLR [PREPRINT], VERIFIED: "C/C++ dominated, being the focus of 91 papers. Java was the target in 4 papers while only 3 studies addressed multiple languages". No Python-specific count is given. - [arXiv PDF](https://arxiv.org/pdf/2506.10280)
- Khalal et al. 2026 [PREPRINT], VERIFIED: "Figure 7b confirms a strong bias toward C and C++ ... many conclusions implicitly reflect C/C++-specific signals ... and may not transfer to managed or scripting languages. Java and Python appear only sporadically, while PHP, Ruby, Go, and Solidity are rarely represented". - [arXiv PDF](https://arxiv.org/pdf/2604.00079)
- Zhang, Yang, Su, Weyssow, Nguyen, Bui, Kang, Li, Ouh, Shar, Lo. "Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection." arXiv:2503.01449, Mar 2025 [PREPRINT], VERIFIED:
  - "Existing research primarily focuses on evaluating LLMs using C/C++ datasets."
  - Dataset: "8,260 vulnerable functions in Python, 7,505 in Java, and 28,983 in JavaScript".
  - Result: "In Python and Java, the results for all LLMs are unsatisfactory, with the best LLM (i.e., CodeQwen1.5 and DeepSeek-Coder) achieving F1 scores of only 0.200 and 0.225, respectively."
  - Imbalance: "the training dataset is highly imbalanced".
  - Source: [arXiv PDF](https://arxiv.org/pdf/2503.01449)

**Multilingual benchmark where Python scores lowest: Shu, Fu, Yu, Wang, Tantithamthavorn, Chen, Kamei. "Evaluating Large Language Models for Multilingual Vulnerability Detection at Dual Granularities." arXiv:2506.07503 (v1 9 Jun 2025, v2 10 Mar 2026). [PREPRINT], no venue in arXiv metadata. VERIFIED** (v2 full text).
- Quote: "the best-performing LLM achieves its highest function-level accuracy (0.8082) with Go and its lowest (0.6626) with Python." GPT-4o per-language accuracy: C 0.7557, C# 0.7273, C++ 0.7901, Go 0.8082, JavaScript 0.6642, Java 0.7169, Python 0.6626. - [arXiv PDF v2](https://arxiv.org/pdf/2506.07503v2)
- Caveats:
  - The claim is for **GPT-4o (instruction-tuned + few-shot), metric = accuracy**.
  - Python is only 0.0016 below JavaScript.
  - At line level GPT-4o's lowest F1 is C# (0.4348), not Python (0.6055).
  - Source: [arXiv PDF v2](https://arxiv.org/pdf/2506.07503v2)
- Data (REEF, function-level, balanced): C 3,056; C++ 1,792; C# 427; Go 2,905; Java 3,235; JavaScript 5,468; **Python 3,282**. In REEF Python is **not** much smaller than C or C++ (computed: C + C++ = 4,848, or 1.5x). - [arXiv HTML v2](https://arxiv.org/html/2506.07503v2)

**Counter-evidence to "Python is always small": CleanVul.** Li, Zhang, Widyasari, et al. (16 authors). "CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics." arXiv:2411.17274 (v7 Sep 2025). [PREPRINT]; the PDF header shows an unfilled ACM template, so the venue is unconfirmed. VERIFIED.
- VulSifter "processed code changes across multiple programming languages: 26,423 Java, 6,591 Python, 5,578 C, 4,000 JavaScript, 312 C#, and 125 C++ changes". The final CleanVul has 8,198 functions at threshold 3.
- The abstract reports noise in existing datasets of "typically 40% to 75%".
- Source: [arXiv PDF](https://arxiv.org/pdf/2411.17274)

### Inferences
- Defensible size comparison for the paper:
  - The Python target (SVEN, 760 functions, 380 vulnerable) is about **310x smaller** than the C/C++ source PrimeVul (235,768 functions). It has about **18x fewer vulnerable functions** (380 vs 6,968). All computed from the two primary papers.
  - Even the largest dedicated Python benchmark (PyVul, 2,082 vulnerable functions, 2025 preprint) is under a third of PrimeVul's vulnerable count.
- SVEN's Python half is CWE-narrow: 4 web/injection CWEs (089, 078, 022, 079). The C/C++ half holds the memory-safety CWEs. This matters for any claim about which CWEs transfer.
- "Python is the hardest language" should **not** be written as a finding. 2506.07503 supports only "lowest GPT-4o function-level accuracy, by 0.0016". 2503.01449 puts Python and Java together as the weak pair.

### Gaps
- CVEfixes' own paper (arXiv v1) has no per-language function table that I could find. The Python counts above come from two derivative papers, and they disagree: 2,775 vulnerable (Al Atiiq) vs 1,360 functions (PyVul) vs 779 vulnerable (MVD). The differences come from CVEfixes version and filtering. Cite the derivative paper that matches the claim, never "CVEfixes" alone.
- Other Python sets seen only in snippets (UNVERIFIED): PyCode-Vul on Kaggle (17,811 functions, 7,899 vulnerable, per search snippet; no paper read); DetectVul (FGCS 2024, statement-level Python; paywalled); RepoPairBench (100 Python pairs, snippet only).

---

## Q3. Transfer from a data-rich source (project, domain, language) as a recognised answer to scarcity (claim c)

### Takeaway
Cross-project and cross-domain transfer for VD is a well-established line with many verified sources: Lin TII 2018, CD-VulD TDSC 2022, Dual-GD-DDAN PAKDD 2020, the max-margin TOSEM 2024 paper, CPVD TSE 2023, MNCRI EMNLP 2023 and ZSVulD EMSE 2025. All of them explicitly motivate transfer by the lack of target labels. Cross-**language** VD is newer and thinner: IRC-CLVul (Electronics 2023, synthetic Juliet only), BABEL (ICSME 2024), MSVD / VDMAF (IST 2025), CLMDA (KBS 2026), and joint multilingual training (MVD, MulVuln; preprints). Almost all cross-project work stays within C/C++ projects (FFmpeg, LibPNG, Chromium, Qemu).

### Cited Findings

**Cross-project / cross-domain (all C/C++ projects)**
- **Lin et al., TII 2018** (bib in Q1). PARTIAL. "This paper addresses this cold-start problem of machine learning, by learning rich features that generalize across similar projects." "the neural representation obtained from existing software projects is then transferred to the new project to enable early vulnerability detection even with a small set of training labels." - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TII.2018.2821768)
- **Liu, Lin, Qu, Zhang, De Vel, Montague, Xiang. "CD-VulD: Cross-Domain Vulnerability Discovery Based on Deep Domain Adaptation." IEEE TDSC 19(1):438-451, Jan 2022 (early access 2020). DOI 10.1109/TDSC.2020.2984505. PARTIAL** (abstract; no numbers in abstract).
  - "in practice, the test data often differs from the training data in terms of distribution because they are from different projects or they differ in the types of vulnerability."
  - Uses "the metric transfer learning framework (MTLF) ... by minimizing the distribution divergence between the source domain and the target domain". Claims it "outperforms the state-of-the-art vulnerability detection approaches by a wide margin".
  - Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TDSC.2020.2984505); bib per [Crossref](https://api.crossref.org/works?query.bibliographic=CD-VulD)
- **Nguyen et al., PAKDD 2020** (bib in Q1). VERIFIED. Multimedia projects are the source and image projects the target. On FFmpeg to LibPNG, Dual-GD-DDAN reaches F1 88.89% (recall 100%, precision 80%) vs DDAN 84.21%. On VLC to LibPNG it reaches 71.43% vs DDAN 70.59%. - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206170/)
- **Nguyen et al., TOSEM 2024** (bib in Q1). PARTIAL. F1 +1.83% to +6.25% over the second-best method, "from imbalanced labeled into imbalanced unlabeled projects". - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3664602)
- **Zhang, Liu, Xin, Yao. "CPVD: Cross Project Vulnerability Detection Based on Graph Attention Network and Domain Adaptation." IEEE TSE 49(8):4152-4168, Aug 2023. DOI 10.1109/TSE.2023.3285910. PARTIAL** (abstract).
  - "Vulnerability annotation in large-scale software code is quite tedious and challenging, which requires domain experts to spend a lot of time annotating."
  - Motivating challenge: "learning to predict the vulnerability labels of another item quickly using one item with rich vulnerability labels."
  - Results on chr_deb, qemu, libav and sard: F1 of 70.2%, 81.1%, 59.7% and 78.1%; AUC of 88.4%, 86.3%, 85.2% and 88.6%.
  - Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TSE.2023.3285910)
- **Du, Zhou, Kuang, Zhao, Zhai. "Joint Geometrical and Statistical Domain Adaptation for Cross-domain Code Vulnerability Detection" (MNCRI). EMNLP 2023, pp. 12791-12800. DOI 10.18653/v1/2023.emnlp-main.788. PARTIAL** (abstract).
  - "a detector trained on a label-rich source domain fails to provide accurate prediction on new or unseen target domains due to the lack of labeled training data on target domains."
  - Warns about "negative transfer" and "excessive alignment".
  - Source: [ACL Anthology](https://aclanthology.org/2023.emnlp-main.788/)
- **Haque, Ali, McClean, Khan. "A zero-shot framework for cross-project vulnerability detection in source code" (ZSVulD). Empirical Software Engineering 31(1), article 3, published online 29 Oct 2025. DOI 10.1007/s10664-025-10749-4. PARTIAL** (abstract; no numbers in abstract).
  - Models "struggle to generalise across projects due to variations in coding styles, feature distributions, and the absence of labelled target data".
  - Uses CodeBERT embeddings plus iterative pseudo-labelling, on Devign and REVEAL. It reports higher recall/F1/F2 than existing methods without giving values in the abstract.
  - Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s10664-025-10749-4); bib per [Crossref](https://api.crossref.org/works/10.1007/s10664-025-10749-4)

**Cross-language VD**
- **Lei, Xue, Wang, Liu. "IRC-CLVul: Cross-Programming-Language Vulnerability Detection with Intermediate Representations and Combined Features." Electronics 12(14):3067, 2023. DOI 10.3390/electronics12143067. PARTIAL** (abstract). [VENUE?]: MDPI Electronics is a mid-tier, high-volume venue.
  - Converts C, C++ and Java to LLVM-IR, then uses a Random Forest.
  - "We conducted experiments on 85,811 samples from the Juliet test suite in C, C++, and Java. The results show that our method improved the accuracy by 7% compared with the two baseline algorithms, and the F1 score showed a 12% increase." The data is synthetic, and there is no Python.
  - Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/electronics12143067)
- **BABEL: Li, Lin, Zheng, He, Liao, Wang (Xia Li, Yuhang Lin, Yongqiang Zheng, Junyi He, Rihu Liao, Junlang Wang). "BABEL: A Novel Software Vulnerability Detection Framework for Breaking Language Barriers." 2024 IEEE ICSME, pp. 599-611, 6 Oct 2024. DOI 10.1109/ICSME58944.2024.00060. IEEE Xplore document 10795082. Code: github.com/gdufsnlp/BABEL. PARTIAL** (abstract + README; the Xplore page returned empty to the fetcher).
  - Abstract: "we present a novel detection framework that can identify software vulnerabilities in a programming language-agnostic manner without relying on code parsers. We conduct extensive experiments on the CodeXGlue,Reveal, and FUNDED public datasets". It claims "superior accuracy and F1 scores across various tasks, including those in language-independent settings".
  - The README's cross-language command trains on `./dataset/codexglue/train.jsonl` (C) and tests on `./dataset/funded/JAVA/CWE-074/all.jsonl`, i.e. C to Java. No Python target is shown.
  - Sources: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICSME58944.2024.00060), [Crossref](https://api.crossref.org/works?query.bibliographic=A+Novel+Software+Vulnerability+Detection+Framework+for+Breaking+Language+Barriers), [GitHub README](https://github.com/gdufsnlp/BABEL)
- **Cao, Dong. "Multi-source cross-domain vulnerability detection based on code pre-trained model" (MSVD). Information and Software Technology 185:107764, Sept 2025. DOI 10.1016/j.infsof.2025.107764. UNVERIFIED content** (bib VERIFIED via [Crossref](https://api.crossref.org/works?query.bibliographic=Multi-source+cross-domain+vulnerability+detection+based+on+code+pre-trained+model); ScienceDirect and ResearchGate returned 403).
  - Search-snippet wording (re-check before quoting): "Labeled data is usually concentrated in a few software projects and programming languages". MSVD combines fine-tuning with adversarial multi-source domain adaptation and reports gains "in the cross-language scenario by 2.95% ... ∼57.83%".
  - Source languages unknown to me. - [ScienceDirect (403)](https://www.sciencedirect.com/science/article/abs/pii/S095058492500103X)
- **Li, Luo, Wu, Zheng. "VDMAF: Cross-language source code vulnerability detection using multi-head attention fusion." Information and Software Technology 183:107739, July 2025. DOI 10.1016/j.infsof.2025.107739.** Bib VERIFIED via [Crossref](https://api.crossref.org/works?query.bibliographic=VDMAF); content UNVERIFIED (paywalled; languages unknown).
- **Zou, Jiang, Zhang, Wang, Xue, Luan. "CLMDA: Cross language vulnerability detection based on multimodal learning and domain adaptation." Knowledge-Based Systems 336:115336, March 2026. DOI 10.1016/j.knosys.2026.115336.** Bib VERIFIED via [Crossref](https://api.crossref.org/works?query.bibliographic=CLMDA); content UNVERIFIED (snippet: F1 +14.26% to +55.32%; languages unknown).
- **MVD** (bib in Q2) [PREPRINT], VERIFIED.
  - "a key limitation of these techniques is their primary focus on a single programming language, such as C/C++".
  - "Further, there appears to be an oversight in harnessing the synergies of vulnerability knowledge across varied languages".
  - Joint training over six languages beats single-language SOTA "by 83.7% to 193.6% in PR-AUC". It "detects vulnerabilities well for new languages without compromising the detection performance of previously trained languages" through incremental learning.
  - Source: [arXiv abs](https://arxiv.org/abs/2412.06166)
- **MulVuln: Nguyen, Nepal, Yuan, Wu, Chen, Rudolph. arXiv:2510.04397, Oct 2025. [PREPRINT]. PARTIAL.** Learns shared plus language-specific knowledge on REEF (7 languages, 4,466 CVEs, 30,987 patches). F1 is "1.45% to 23.59%" above 13 baselines. - [arXiv abs](https://arxiv.org/abs/2510.04397)

### Inferences
- Nearly every cross-project paper above motivates transfer with the same argument: expert labels are expensive and the target project has few or none. The citing project's argument has a strong precedent.
- Most classic VD transfer is **unsupervised domain adaptation** (no target labels): CD-VulD, Dual-GD-DDAN, CPVD, MNCRI, ZSVulD. The citing project instead does **sequential pre-train then supervised fine-tune** on a small labeled target. The closer analogue is the Lin TII 2018 "small set of training labels" setting and the low-resource-PL fine-tuning work in Q4. Say this explicitly so reviewers do not expect a DA comparison.
- Verified cross-language VD sources/targets are C, C++ and Java (IRC-CLVul; BABEL README) or joint multilingual pools (MVD, MulVuln). None of the verified ones is a "C/C++ + JS to small Python target" design.

### Gaps
- Numbers for CD-VulD, ZSVulD and BABEL are behind paywalls; only their abstracts' qualitative claims are verified.
- MSVD, VDMAF and CLMDA: languages and exact numbers unverified. **These are the three papers most likely to contain a Python target and must be read before claiming novelty** (see Q6).

---

## Q4. Low-resource programming languages with code models: does source-language / multilingual training help a small target language?

### Takeaway
Yes, for non-VD tasks. Ahmed & Devanbu (ICSE 2022) show multilingual fine-tuning helps most for the lowest-resource language: +17.7% BLEU-4 for Ruby vs +2.5% for Python. They also show cross-language training (e.g. train on Python, test on Ruby) can beat same-language training. Chen et al. (ICPC 2022) confirm that combined multilingual fine-tuning is best for Ruby. In both papers the "low-resource" language is Ruby; Python is the **high-resource** language. No equivalent verified study exists for VD.

### Cited Findings
- **Ahmed, Devanbu. "Multilingual training for software engineering." ICSE 2022 (44th ICSE), pp. 1443-1455. DOI 10.1145/3510003.3510049. arXiv:2112.02043. VERIFIED** (arXiv full text).
  - Abstract: "For some languages (e.g., Ruby) labeled data is less abundant; in others (e.g., JavaScript) the available data maybe more focused on some application domains, and thus less diverse."
  - Contribution (2): "cross-language training (e.g., train on Python, test on Ruby) can sometimes lead to better performance than same-language training."
  - "With CodeBERT, multilingual fine-tuning gains 2.5%-17.5% over monolingual fine-tuning, for all languages, yielding a 6.90% overall improvement (4.48% weighted improvement)". Null hypothesis rejected for all six languages with a one-sided Wilcoxon test.
  - "BLEU-4 gains are higher for low-resource language (e.g., 17.7% for Ruby), and lower for high-resource languages (e.g., 2.5% for Python), as expected." The paper itself gives both 17.5% and 17.7%; quote whichever sentence is used, verbatim.
  - Sources: [arXiv abs](https://arxiv.org/abs/2112.02043), [arXiv PDF](https://arxiv.org/pdf/2112.02043), bib per [Crossref](https://api.crossref.org/works?query.bibliographic=Multilingual+training+for+software+engineering&query.author=Devanbu)
- **Chen, Fard, Lo, Bryksin. "On the transferability of pre-trained language models for low-resource programming languages." ICPC 2022 (30th ICPC), pp. 401-412. DOI 10.1145/3524610.3527917. arXiv:2204.09653. VERIFIED** (arXiv full text).
  - Ruby was chosen "because it is highly ranked among low-resource languages".
  - "for all the PLMs, fine-tuning on the combined dataset gives the best performance" (code summarization, and likewise for code search).
  - Monolingual PLMs "outperformed CodeBERT in MRR between 5% and 35.1%, and GraphCodeBERT in MRR between 2.3% and 32.6%". Mann-Whitney p = 0.00256.
  - Abstract: multilingual PLMs have a "lower Performance-to-Time Ratio" than monolingual PLMs.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2204.09653)

### Inferences
- These two papers support the **mechanism** (knowledge from high-resource languages helps a small target language) but **not for VD and not with Python as the target**. When citing them, the paper should say "analogous evidence from code summarization / search", not "evidence for VD".

### Gaps
- No verified VD-specific study measures how much a source language helps a low-resource VD target as a function of target size. MVD's incremental-learning result is the closest, but it is a preprint and its Python target is not small.

---

## Q5. Data augmentation / generation as an alternative answer to scarcity (brief)

### Takeaway
VulGen (ICSE 2023) and VGX (ICSE 2024) are the standard citations. Both frame generation as the answer to "lack of large and quality sets of labeled vulnerable program samples", and both report large F1 gains for downstream detectors. Auto-labeling (Le & Babar 2024) is a third answer, with a noise trade-off.

### Cited Findings
- **Nong, Ou, Pradel, Chen, Cai. "VULGEN: Realistic Vulnerability Generation Via Pattern Mining and Deep Learning." ICSE 2023, pp. 2527-2539. DOI 10.1109/ICSE48619.2023.00211. PARTIAL** (abstract).
  - "Building new, powerful data-driven defenses against prevalent software vulnerabilities needs sizable, quality vulnerability datasets".
  - Generated samples gave "substantial performance improvements for two SOTA DL-based vulnerability detectors (by up to 31.8% higher in F1)".
  - Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICSE48619.2023.00211)
- **Nong, Fang, Yi, Zhao, Luo, Chen, Cai. "VGX: Large-Scale Sample Generation for Boosting Learning-Based Software Vulnerability Analyses." ICSE 2024 (46th ICSE), pp. 1-13. DOI 10.1145/3597503.3639116. arXiv:2310.15436. PARTIAL** (abstract; the PDF text was downloaded but only grepped).
  - "Accompanying the successes of learning-based defensive software vulnerability analyses is the lack of large and quality sets of labeled vulnerable program samples".
  - "VGX generated 150,392 vulnerable samples". Adding 10% of them gave detection "19.15-330.80% higher F1".
  - The paper says its methodology "is not limited to a particular programming language" when given AST and value-flow parsers.
  - Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3597503.3639116)
- **Le & Babar, ESEM 2024** (Q1): auto-labeled D2A data is 50+% noisy but still lifts MCC by up to 22%. - [arXiv abs](https://arxiv.org/abs/2407.17803)

### Inferences
- Augmentation and transfer are complementary answers to the same scarcity. A related-work sentence can cite VulGen/VGX as "the generation route" and position cross-language pre-training as "the transfer route".

### Gaps
- I did not verify which language VulGen and VGX evaluate on (believed to be C; not confirmed). Do not say "C only" without checking.

---

## Q6. Is there any published work transferring from C/C++ or JavaScript to Python for VD? (re-check 2025-2026)

### Takeaway
Not quite "none" any more. One **April 2026 preprint** (arXiv:2604.27714) fine-tunes 8B LLMs on **C/C++ Juliet** and evaluates **zero-shot on Python** (BenchmarkPython) and Java. That is C/C++ to Python cross-language VD, but it is synthetic, uses no target fine-tuning, and studies false-positive behaviour. I found **no** published work that pre-trains on **real-world C/C++ and/or JavaScript** VD data and then **fine-tunes on a small labeled real-world Python set** (such as SVEN). That gap still holds, with the caveat that three paywalled cross-language papers (MSVD, VDMAF, CLMDA) could not be checked for a Python target.

### Cited Findings
- **Chen, Wang, Qin, Wang, Wu, Liu (Maofei Chen, Laifu Wang, Yue Qin, Yuan Wang, Bo Wu, Dongxin Liu). "How Code Representation Shapes False-Positive Dynamics in Cross-Language LLM Vulnerability Detection." arXiv:2604.27714, 30 Apr 2026. [PREPRINT]. VERIFIED** (full text).
  - Models "fine-tuned on C/C++ data from the NIST Juliet Test Suite (v1.3) and evaluated on Java (OWASP Benchmark v1.2) and Python (BenchmarkPython v0.1)".
  - Training size: "∼3,100 training samples" (pilot) and "∼70,700" (full).
  - Python target: "1,108-sample subset (420 vulnerable, 688 benign; 14 CWEs)".
  - Key numbers (Qwen3-8B on OWASP Java): FPR "0.763 zero-shot, 0.866 pilot, 1.000 full-scale" while "F1 remains stable (0.637-0.688)". The "text fine-tuning encodes C/C++-specific API names and syntactic idioms as vulnerability triggers that fire indiscriminately on target-language code".
  - "On BenchmarkPython the AST probe yields FPR=0.554, within 2.9 percentage points of the Java result".
  - Source: [arXiv abs](https://arxiv.org/abs/2604.27714), [arXiv PDF](https://arxiv.org/pdf/2604.27714)
- **MVD** (Q2/Q3) [PREPRINT]: joint multilingual training including Python (779 vulnerable Python functions). Incremental learning adds new languages. This is multi-source joint training, not source-to-Python fine-tuning. - [arXiv PDF](https://arxiv.org/pdf/2412.06166)
- **MulVuln** (Q3) and **Shu et al. 2506.07503** / **"A Preliminary Study of Large Language Models for Multilingual Vulnerability Detection"** (arXiv:2505.07376, [PREPRINT]) evaluate or train on all 7 REEF languages together. They are not transfer experiments.
  - 2505.07376 abstract: "existing methods are predominantly limited to specific programming languages, restricting their applicability in multilingual settings" and "This work represents an initial step toward exploring PLMs and LLMs for cross-language vulnerability detection".
  - Sources: [arXiv abs 2505.07376](https://arxiv.org/abs/2505.07376), [arXiv abs 2510.04397](https://arxiv.org/abs/2510.04397)
- **BABEL** (ICSME 2024): the verified cross-language demo is C (CodeXGLUE) to Java (FUNDED). - [GitHub README](https://github.com/gdufsnlp/BABEL)
- **IRC-CLVul** (Electronics 2023): C, C++ and Java on Juliet only. - [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/electronics12143067)
- **Farr et al. "Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection."** arXiv:2506.10104, Jun 2025 [PREPRINT], PARTIAL. It is Python-targeted, but "cross-domain" here means across CWE categories within prompting, not across languages. - [arXiv abs](https://arxiv.org/abs/2506.10104)
- **Humran, Sonmez. "Code Vulnerability Detection Across Different Programming Languages with AI Models."** arXiv:2508.11710 [PREPRINT] [VENUE?]. The abstract is vague ("accuracy greater than 97%") and has no transfer protocol. Not usable. - [arXiv abs](https://arxiv.org/abs/2508.11710)
- The project's earlier notes already excluded "Cross-Language Transfer Learning for Detecting Vulnerabilities in LLM-Generated Code" (*Global Media and Social Sciences Research Journal*, 2/2026) for an out-of-field venue. Keep it excluded [VENUE?].

### Inferences
- Safe novelty wording: "To our knowledge, no prior work pre-trains on real-world C/C++ and JavaScript/Java vulnerability data and fine-tunes on a small, manually curated Python VD set. The closest work either trains jointly on all languages (MVD, MulVuln), transfers among C/C++/Java only (IRC-CLVul, BABEL), or evaluates zero-shot transfer from synthetic C/C++ Juliet to a synthetic Python benchmark (arXiv:2604.27714)."
- arXiv:2604.27714 is also a **useful warning to cite**: C/C++-trained models fire on C/C++ surface cues when applied to Python, and FPR climbs toward 1.0 with more source training. That is directly relevant to whether Phase 1 on PrimeVul helps or hurts a Python target, and it argues for reporting FPR or threshold-free metrics alongside F1.

### Gaps
- MSVD (IST 2025), VDMAF (IST 2025) and CLMDA (KBS 2026) are cross-language VD papers whose language pairs I could not read (paywall / 403). **Read these three before finalising the novelty sentence.** If any has a Python target, the wording above must be narrowed further.
- The review of 2025-2026 used web search plus arXiv pages. Very recent conference papers (e.g. ICSE/FSE/ASE 2026 proceedings not yet on arXiv) may be missed.
