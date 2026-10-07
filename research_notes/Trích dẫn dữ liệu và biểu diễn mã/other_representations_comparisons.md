# Non-graph code representations for vulnerability detection (VD), and graph-vs-other comparisons

_Status legend: **VERIFIED** = number/quote read in the primary full text (arXiv PDF or author PDF); **PARTIAL** = only the abstract/metadata page of the primary source was read; **UNVERIFIED** = only seen in a secondary source (search summary, other paper). Bibliographic fields (authors, pages, DOI) were checked against Crossref and the arXiv abstract page unless noted; the Crossref links point to the DOI record, which for Siow et al., VELVET, PrimeVul and Vul-LMGNNs was located through a Crossref bibliographic query (`api.crossref.org/works?query.bibliographic=...`). All sources retrieved 2026-09-28. Per project style, long dashes inside quotes are rendered as "-"; everything else in quotes is verbatim (pdftotext output; ligatures/spacing normalised)._

## Q1. Token / sequence representations (Russell et al. 2018; VulDeePecker; SySeVR; LineVul)

### Takeaway
The early "sequence" VD models are of two kinds: truly graph-free lexed token streams (Russell et al.; LineVul's BPE sequence into a CodeBERT-initialised RoBERTa with a 512-token block), and sequence models over **dependency-based program slices** (VulDeePecker code gadgets, SySeVR SyVC/SeVC), which are graph-derived even though the network sees a sequence. LineVul displaced the graph model IVDetect on Big-Vul (160%-379% higher function-level F1, per its abstract), but that benchmark was later shown to be heavily overestimated (see Q5, PrimeVul).

### Cited Findings

**1. Russell et al., ICMLA 2018 - VERIFIED**
- Bib: Rebecca L. Russell, Louis Kim, Lei H. Hamilton, Tomo Lazovich, Jacob A. Harer, Onur Ozdemir, Paul M. Ellingwood, Marc W. McConley. "Automated Vulnerability Detection in Source Code Using Deep Representation Learning." *2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)*, Orlando, FL, Dec 2018, pp. 757-762. DOI 10.1109/ICMLA.2018.00120. arXiv:1807.04320.
- Read: [arXiv abs](https://arxiv.org/abs/1807.04320), [arXiv PDF](https://arxiv.org/pdf/1807.04320), [Crossref](https://api.crossref.org/works/10.1109/ICMLA.2018.00120).
- Representation: custom C/C++ lexer to a small token vocabulary; CNN or RNN over the token sequence; learned features optionally fed to a random forest (RF). Quote: "Our lexer was able to reduce C/C++ code to representations using a total vocabulary size of only 156 tokens." - [arXiv PDF](https://arxiv.org/pdf/1807.04320)
- Labels on the Debian/GitHub portion come from static analyzers: "we compiled a vast dataset of millions of open-source functions and labeled it with carefully-selected findings from three different static analyzers" - [arXiv abs](https://arxiv.org/abs/1807.04320)
- Table III (Debian + GitHub test; columns PR AUC / ROC AUC / MCC / F1): BOW+RF 0.459 / 0.883 / 0.462 / 0.498; RNN 0.465 / 0.896 / 0.501 / 0.532; CNN 0.467 / 0.897 / 0.509 / 0.540; RNN+RF 0.498 / 0.899 / 0.523 / 0.552; CNN+RF 0.518 / 0.904 / 0.536 / 0.566. Table IV (SATE IV Juliet): BOW+RF 0.890 / 0.913 / 0.607 / 0.786; CNN 0.944 / 0.954 / 0.698 / 0.840; CNN+RF 0.916 / 0.936 / 0.672 / 0.824. - [arXiv PDF](https://arxiv.org/pdf/1807.04320)
- On the bag-of-words baseline: "the classifier exploits label correlations with (1) indicators of the source length and complexity and (2) combinations of calls" - [arXiv PDF](https://arxiv.org/pdf/1807.04320)

**2. VulDeePecker, NDSS 2018 - VERIFIED**
- Bib: Zhen Li, Deqing Zou, Shouhuai Xu, Xinyu Ou, Hai Jin, Sujuan Wang, Zhijun Deng, Yuyi Zhong. "VulDeePecker: A Deep Learning-Based System for Vulnerability Detection." *Proceedings 2018 Network and Distributed System Security Symposium (NDSS 2018)*. DOI 10.14722/ndss.2018.23158. arXiv:1801.01681. (Page numbers not returned by Crossref.)
- Read: [arXiv abs](https://arxiv.org/abs/1801.01681), [arXiv PDF](https://arxiv.org/pdf/1801.01681), [Crossref](https://api.crossref.org/works/10.14722/ndss.2018.23158).
- Representation: "a code gadget is a number of (not necessarily consecutive) lines of code that are semantically related to each other" (built from program slices around library/API calls), vectorised and classified by a BLSTM; "The number of tokens in the vector representation of code gadgets is set to 50". - [arXiv PDF](https://arxiv.org/pdf/1801.01681)
- Table II (FPR / FNR / TPR / P / F1, %): BE-ALL (buffer errors) 2.9 / 18.0 / 82.0 / 91.7 / 86.6; RM-ALL (resource management) 2.8 / 4.7 / 95.3 / 94.6 / 95.0; HY-ALL (both) 5.1 / 16.1 / 83.9 / 86.9 / 85.4. - [arXiv PDF](https://arxiv.org/pdf/1801.01681)
- Real-world claim: applied to Xen, Seamonkey and Libav it "detect[s] 4 vulnerabilities, which are not reported in the National Vulnerability Database but were 'silently' patched". - [arXiv abs](https://arxiv.org/abs/1801.01681)

**3. SySeVR, IEEE TDSC - VERIFIED**
- Bib: Zhen Li, Deqing Zou, Shouhuai Xu, Hai Jin, Yawei Zhu, Zhaoxuan Chen. "SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities." *IEEE Transactions on Dependable and Secure Computing* 19(4):2244-2258, 2022 (Crossref publication date 2022-07-01; the DOI string carries 2021). DOI 10.1109/TDSC.2021.3051525. arXiv:1807.06756.
- Read: [arXiv abs](https://arxiv.org/abs/1807.06756), [arXiv PDF](https://arxiv.org/pdf/1807.06756), [Crossref](https://api.crossref.org/works/10.1109/TDSC.2021.3051525).
- Representation: "SyVCs reflect vulnerability syntax characteristics, and SeVCs extend SyVCs to accommodate the semantic information induced by data dependency and control dependency." Sequences are cut to a fixed length centred on the SyVC ("we retain 249 consecutive symbols immediately left to the SyVC and 250 symbols immediately right to the SyVC"), an early explicit truncation policy. - [arXiv PDF](https://arxiv.org/pdf/1807.06756)
- Finding: "semantic information induced by data dependency and control dependency can reduce the false-negative rate by 30.4% on average." "Bidirectional RNNs, especially Bidirectional Gated Recurrent Unit (BGRU), are more effective than unidirectional RNNs and CNNs, which are more effective than DBNs and shallow learning models." - [arXiv PDF](https://arxiv.org/pdf/1807.06756)
- Table 6 (FPR / FNR / A / P / F1 / MCC, %): SySeVR-BGRU 1.4 / 5.6 / 98.0 / 90.8 / 92.6 / 90.5; VulDeePecker 2.5 / 41.8 / 92.2 / 78.0 / 66.6 / 64.9; Checkmarx 20.8 / 56.8 / 72.9 / 30.9 / 36.1 / 33.0; Flawfinder 21.6 / 70.4 / 69.8 / 22.8 / 25.7 / 22.1; RATS 21.5 / 85.3 / 67.2 / 12.8 / 13.7 / 12.6; VUDDY 4.3 / 90.1 / 71.2 / 47.7 / 16.4 / 15.2. - [arXiv PDF](https://arxiv.org/pdf/1807.06756)

**4. LineVul, MSR 2022 - PARTIAL (abstract + code README; full text not accessible)**
- Bib: Michael Fu, Chakkrit Tantithamthavorn. "LineVul: A Transformer-based Line-Level Vulnerability Prediction." *Proceedings of the 19th International Conference on Mining Software Repositories (MSR '22)*, Pittsburgh, May 23-24 2022, pp. 608-620. DOI 10.1145/3524842.3528452. No arXiv version.
- Read: [Monash research portal](https://research.monash.edu/en/publications/linevul-a-transformer-based-line-level-vulnerability-prediction/), [GitHub README](https://raw.githubusercontent.com/awsm-research/LineVul/main/README.md), [Crossref](https://api.crossref.org/works/10.1145/3524842.3528452).
- Representation: token sequence of the whole function into a CodeBERT-based transformer ("CodeBERT, as employed in the recently proposed LineVul" - [StagedVulBERT](https://arxiv.org/pdf/2410.05766)); the README training/eval commands use `--block_size 512`. - [GitHub README](https://raw.githubusercontent.com/awsm-research/LineVul/main/README.md)
- Quote (abstract): "Recently, IVDetect (a graph-based neural network) is proposed to predict vulnerabilities at the function level. Yet, the IVDetect approach is still inaccurate and coarse-grained... Through an empirical evaluation of a large-scale real-world dataset with 188k+ C/C++ functions, we show that LINEVUL achieves (1) 160%-379% higher F1-measure for function-level predictions; (2) 12%-25% higher Top-10 Accuracy for line-level predictions; and (3) 29%-53% less Effort@20%Recall than the baseline approaches" - [Monash portal](https://research.monash.edu/en/publications/linevul-a-transformer-based-line-level-vulnerability-prediction/)
- Independent confirmation of the 512 limit (VERIFIED in a third-party paper): "the best-performing transformer-based method, LineVul, for vulnerability detection only accepts up to 512 tokens [10], which cannot detect the vulnerable example in Fig. 1 because the vulnerable statement exceeds the first 512 tokens." - [StagedVulBERT, arXiv:2410.05766](https://arxiv.org/pdf/2410.05766)

### Inferences
- "Non-graph" is a blurry category: VulDeePecker and SySeVR need data/control dependence to build slices, so their advantage over plain tokens (SySeVR: FNR reduced 30.4% on average by dependence information) is itself evidence that dependence structure matters, delivered through a sequence model.
- Russell et al. show learned token features beat bag-of-words only modestly on real code (Debian/GitHub F1 0.498 BOW+RF vs 0.566 CNN+RF), and their labels are static-analyzer findings, so those numbers measure agreement with analyzers rather than true vulnerability.

### Gaps
- LineVul's absolute F1 values (often cited as ~0.91 vs IVDetect ~0.35 on Big-Vul) could not be read in the primary PDF (ACM/ResearchGate blocked); only the relative 160%-379% claim is verified at abstract level.
- Page ranges for VulDeePecker (NDSS) were not returned by Crossref.

## Q2. Pre-trained transformer encoders as VD backbones, and the 512-token limit

### Takeaway
CodeBERT, GraphCodeBERT, UniXcoder, CodeT5 and CodeT5+ are general code PLMs; the CodeBERT/GraphCodeBERT/UniXcoder papers do not report VD at all, and CodeT5/CodeT5+ only report CodeXGLUE Devign "defect detection" accuracy (62.1-66.7%). CodeBERT, GraphCodeBERT and CodeT5 were pre-trained at 512 tokens (UniXcoder at 1024). Truncation is not a corner case: 41.7% (Devign), 51.8% (DiverseVul) and 64.9% (PrimeVul) of vulnerable C/C++ functions exceed 512 CodeT5 tokens, and in PrimeVul 27% of vulnerable/patched pairs become identical after truncation while keeping opposite labels.

### Cited Findings

**5. CodeBERT, Findings of EMNLP 2020 - VERIFIED**
- Bib: Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, Ming Zhou. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 1536-1547. DOI 10.18653/v1/2020.findings-emnlp.139. arXiv:2002.08155.
- Read: [arXiv abs](https://arxiv.org/abs/2002.08155), [arXiv PDF](https://arxiv.org/pdf/2002.08155), [Crossref](https://api.crossref.org/works/10.18653/v1/2020.findings-emnlp.139).
- Representation: bimodal NL-PL transformer encoder trained with MLM + replaced token detection. Input limit: "We set the max length as 512 and the max training step is 100K." Evaluated in its own paper on code search and documentation generation, not VD. - [arXiv PDF](https://arxiv.org/pdf/2002.08155)

**6. GraphCodeBERT, ICLR 2021 - VERIFIED (arXiv full text; venue from arXiv comment "Accepted by ICLR2021"; OpenReview blocked by a bot challenge; ICLR has no DOI)**
- Bib: Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng, Colin Clement, Dawn Drain, Neel Sundaresan, Jian Yin, Daxin Jiang, Ming Zhou. "GraphCodeBERT: Pre-training Code Representations with Data Flow." ICLR 2021. arXiv:2009.08366 (DOI 10.48550/arXiv.2009.08366).
- Read: [arXiv abs](https://arxiv.org/abs/2009.08366), [arXiv PDF](https://arxiv.org/pdf/2009.08366).
- Representation (a sequence + graph hybrid at pre-training): "we use data flow in the pre-training stage, which is a semantic-level structure of code that encodes the relation of 'where-the-value-comes-from' between variables"; "We set the max length of sequences and nodes as 512 and 128, respectively." Its paper contains no defect/vulnerability-detection experiment (full-text search). - [arXiv PDF](https://arxiv.org/pdf/2009.08366)

**7. UniXcoder, ACL 2022 - VERIFIED**
- Bib: Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, Jian Yin. "UniXcoder: Unified Cross-Modal Pre-training for Code Representation." *Proceedings of the 60th Annual Meeting of the ACL (Volume 1: Long Papers)*, pp. 7212-7225, 2022. DOI 10.18653/v1/2022.acl-long.499. arXiv:2203.03850.
- Read: [arXiv abs](https://arxiv.org/abs/2203.03850), [arXiv PDF](https://arxiv.org/pdf/2203.03850), [Crossref](https://api.crossref.org/works/10.18653/v1/2022.acl-long.499).
- Representation: "To encode AST that is represented as a tree in parallel, we propose a one-to-one mapping method to transform AST in a sequence structure that retains all structural information from the tree." Pre-training length: "we set both the max length of input sequence and batch size as 1024". No VD experiment in the paper. - [arXiv PDF](https://arxiv.org/pdf/2203.03850)
- Conflict to note: Li et al. 2025 (item 13) list `unixcoder-base-nine` with 512 "Max Position Embeddings", whereas Safdar et al. 2025 (item 14) fine-tune UniXcoder at 1024. - [RevisitVD](https://arxiv.org/pdf/2507.16887); [Safdar et al.](https://arxiv.org/pdf/2508.16625)

**8. CodeT5, EMNLP 2021 - VERIFIED**
- Bib: Yue Wang, Weishi Wang, Shafiq Joty, Steven C. H. Hoi. "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation." *Proceedings of EMNLP 2021*, pp. 8696-8708. DOI 10.18653/v1/2021.emnlp-main.685. arXiv:2109.00859.
- Read: [arXiv abs](https://arxiv.org/abs/2109.00859), [arXiv PDF](https://arxiv.org/pdf/2109.00859), [Crossref](https://api.crossref.org/works/10.18653/v1/2021.emnlp-main.685).
- Representation: encoder-decoder with identifier-aware denoising; "We set the maximum source and target sequence lengths to be 512 and 256, respectively."
- Defect detection (CodeXGLUE, from Devign) accuracy: RoBERTa 61.05, CodeBERT 62.08, PLBART 63.18, CodeT5-small 63.40, CodeT5-base 65.78 (CodeT5-base +multi-task 65.02). - [arXiv PDF](https://arxiv.org/pdf/2109.00859)

**9. CodeT5+, EMNLP 2023 - VERIFIED**
- Bib: Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D. Q. Bui, Junnan Li, Steven C. H. Hoi. "CodeT5+: Open Code Large Language Models for Code Understanding and Generation." *Proceedings of EMNLP 2023*, pp. 1069-1088. DOI 10.18653/v1/2023.emnlp-main.68. arXiv:2305.07922.
- Read: [arXiv abs](https://arxiv.org/abs/2305.07922), [arXiv PDF](https://arxiv.org/pdf/2305.07922), [Crossref](https://api.crossref.org/works/10.18653/v1/2023.emnlp-main.68).
- Table 9, defect detection accuracy (CodeXGLUE/Devign, "more than 27,000 annotated functions in C programming language"): CodeBERT 125M 62.1; CodeGen-multi 350M 63.1; PLBART 140M 63.2; CodeT5 220M 65.8; CodeT5+ 220M 66.1; CodeT5+ 770M 66.7 (GraphCodeBERT and UniXcoder: "-", not reported). - [arXiv PDF](https://arxiv.org/pdf/2305.07922)

**10. Evertz et al., "Chasing Shadows", NDSS 2026 - VERIFIED (arXiv full text; NDSS proceedings pages/DOI not checked)**
- Bib: Jonathan Evertz, Niklas Risse, Nicolai Neuer, Andreas Müller, Philipp Normann, Gaetano Sapia, Srishti Gupta, David Pape, Soumya Shaw, Devansh Srivastav, Christian Wressnegger, Erwin Quiring, Thorsten Eisenhofer, Daniel Arp, Lea Schönherr. "Chasing Shadows: Pitfalls in LLM Security Research." arXiv:2512.09549 (Dec 2025); arXiv comment: "About to appear at NDSS'26".
- Read: [arXiv abs](https://arxiv.org/abs/2512.09549), [arXiv PDF](https://arxiv.org/pdf/2512.09549).
- Table IV, vulnerable functions whose CodeT5-tokenized length exceeds the window: Devign 12,460 functions: >512 5,196 (41.7%), >1024 2,642 (21.2%), >2048 984 (7.9%); DiverseVul 18,945: 9,814 (51.8%), 5,835 (30.8%), 2,747 (14.5%); PrimeVul 6,004: 3,897 (64.9%), 2,594 (43.2%), 1,357 (22.6%); "Average" row 52.8% / 31.7% / 15.0%. - [arXiv PDF](https://arxiv.org/pdf/2512.09549)
- Text quote: "on average, 49.3% of all vulnerable functions across Devign, DiverseVul, and PrimeVul contain more than 512 tokens when tokenized with the CodeT5 tokenizer. At the 1024-token threshold, 29.1% still exceed the limit, and 13.7% surpass 2048 tokens... Notably, these percentages represent a lower bound on the issue; Risse et al. [112] show that vulnerability detection often depends on code outside the function itself". - [arXiv PDF](https://arxiv.org/pdf/2512.09549)
- **Internal inconsistency:** the text (49.3 / 29.1 / 13.7%) does not match the table's Average row (52.8 / 31.7 / 15.0%, which equals the unweighted mean of the three per-dataset rates); the pooled rate from the table counts is 18,907 / 37,409 = 50.5% (my arithmetic). Cite the per-dataset rows.

**11. StagedVulBERT (Jiang et al., arXiv 2024) - VERIFIED (arXiv; peer-reviewed venue not stated on arXiv)**
- Bib: Yuan Jiang, Yujian Zhang, Xiaohong Su, Christoph Treude, Tiantian Wang. "StagedVulBERT: Multi-Granular Vulnerability Detection with a Novel Pre-trained Code Model." arXiv:2410.05766 (Oct 2024).
- Read: [arXiv abs](https://arxiv.org/abs/2410.05766), [arXiv PDF](https://arxiv.org/pdf/2410.05766).
- Quote: "with a 512-token limit, 17% of samples in the big vul dataset are truncated, leading to potential information loss. However, increasing the token limit to 1024 and 2048 reduces truncation to 7% and 2%, respectively". PCL models "are limited by sequence lengths - typically a maximum of 512 tokens. Thus, code functions that exceed the maximum length (i.e., 512) will simply be truncated". Abstract: coarse-grained F1 92.26%, "a 6.58% improvement over the best-performing methods". - [arXiv PDF](https://arxiv.org/pdf/2410.05766)

**12. Steenhoek et al., ICSE 2023, on truncation - VERIFIED** (full bib in Q5 item 20)
- "the transformer models sometimes made predictions without seeing the root cause. This is because the transformer models take a fixed-size input, and some code, sometimes including the root cause, is truncated. Interestingly, those models are still able to correctly predict whether a function is vulnerable with high F1 score." - [arXiv PDF](https://arxiv.org/pdf/2212.08109)

**13. Li et al., "Revisiting Pre-trained Language Models for Vulnerability Detection" (RevisitVD), AsiaCCS 2026 - VERIFIED (arXiv)**
- Bib: Youpeng Li, Weiliang Qi, Xuyu Wang, Fuxun Yu, Xinda Wang. "Revisiting Pre-trained Language Models for Vulnerability Detection." arXiv:2507.16887 (Jul 2025); arXiv comment: "Accepted by the 21st ACM ASIA Conference on Computer and Communications Security (AsiaCCS 2026)".
- Read: [arXiv abs](https://arxiv.org/abs/2507.16887), [arXiv PDF](https://arxiv.org/pdf/2507.16887).
- Truncation creates label noise: "Due to truncation, the visible input portions of the vulnerable and non-vulnerable function appear identical to the Code SLMs. However, their ground-truth labels are opposite... By analyzing 5,480 patch pairs in PrimeVul, we find that 1,473 pairs (27%) have this issue." Slicing fix: "Code SLMs fine-tuned on the sliced training set achieve an average performance improvement of 4% compared to those trained on the original paired training set." Models listed at 512 max positions: codebert-base, graphcodebert-base, unixcoder-base-nine, codet5-base, pdbert-base. - [arXiv PDF](https://arxiv.org/pdf/2507.16887)
- Structure-in-pretraining helps: "PLMs incorporating pre-training tasks designed to capture the syntactic and semantic patterns of code outperform both general-purpose PLMs and those solely pre-trained or fine-tuned on large code corpora." - [arXiv abs](https://arxiv.org/abs/2507.16887)

**14. Safdar et al., "Data and Context Matter" (arXiv 2025, preprint) - VERIFIED (arXiv)**
- Bib: Rijha Safdar, Danyail Mateen, Syed Taha Ali, M. Umer Ashfaq, Wajahat Hussain. "Data and Context Matter: Towards Generalizing AI-based Software Vulnerability Detection." arXiv:2508.16625 (Aug 2025).
- Read: [arXiv abs](https://arxiv.org/abs/2508.16625), [arXiv PDF](https://arxiv.org/pdf/2508.16625).
- Quote: "UniXcoder-Base improved from 84.68% F1 (context window:512) to 94.23% F1 (context window:1024). Similarly, UniXcoder-Base-Nine improved from 88.8% F1 (context window:512) to 94.73% F1 (context window:1024)." (BigVul). Also: "newer transformer architectures, pre-trained on code, dramatically outperform GNN and deep-learning solutions." - [arXiv PDF](https://arxiv.org/pdf/2508.16625)

### Inferences
- The CodeXGLUE Devign accuracy spread across all code PLMs (62.1-66.7%, items 8-9) is small, so "which backbone" is a second-order choice compared with input coverage (items 10-13).
- For a multi-window design (as in MultiVD), items 10 and 13 are the most direct motivation: a single 512-token window misses code in a large fraction of vulnerable functions and turns some vulnerable/fixed pairs into contradictory duplicates.
- The Safdar et al. +9.5 F1 gain from 512 to 1024 is on BigVul, whose scores PrimeVul shows to be inflated (item 21); it indicates direction, not magnitude.

### Gaps
- No primary source found that measures the >512-token rate on **Python** VD data (e.g., SVEN); all verified truncation statistics are C/C++ (Devign, DiverseVul, PrimeVul, Big-Vul). This would have to be measured locally with the same tokenizer.
- StagedVulBERT's vulnerable-only truncation rate (its Fig. 10b) is only in a figure, not in the text read.
- arXiv:2505.17460 ("Learning to Focus: Context Extraction for Efficient Code Vulnerability Detection with Language Models") surfaced in searches about truncation but is **withdrawn** ("Due to fundamental errors in the methodology, I have to withdraw this paper") - do not cite. - [arXiv abs](https://arxiv.org/abs/2505.17460)

## Q3. AST / path representations (code2vec, code2seq, ASTNN, TBCNN)

### Takeaway
All four AST models were introduced and evaluated on non-VD tasks (method naming, summarisation, program classification, clone detection). In the two VD comparisons read here, the AST-path model code2vec is roughly on par with graph models on Devign (accuracy 0.7180 vs GGNN 0.7158; F1 0.7192 vs 0.7344) and is the most seed-stable of nine VD models.

### Cited Findings

**15. code2vec, POPL 2019 - VERIFIED**
- Bib: Uri Alon, Meital Zilberstein, Omer Levy, Eran Yahav. "code2vec: Learning Distributed Representations of Code." *Proceedings of the ACM on Programming Languages* 3(POPL), 2019, pp. 1-29 (Crossref). DOI 10.1145/3290353. arXiv:1803.09473.
- Read: [arXiv abs](https://arxiv.org/abs/1803.09473), [arXiv PDF](https://arxiv.org/pdf/1803.09473), [Crossref](https://api.crossref.org/works/10.1145/3290353).
- Representation: "decomposing code to a collection of paths in its abstract syntax tree, and learning the atomic representation of each path simultaneously with learning how to aggregate a set of them." Original task: method-name prediction, "a relative improvement of over 75%". - [arXiv abs](https://arxiv.org/abs/1803.09473)

**16. code2seq, ICLR 2019 - VERIFIED (arXiv; venue from arXiv comment; ICLR has no DOI)**
- Bib: Uri Alon, Shaked Brody, Omer Levy, Eran Yahav. "code2seq: Generating Sequences from Structured Representations of Code." ICLR 2019. arXiv:1808.01400.
- Read: [arXiv abs](https://arxiv.org/abs/1808.01400), [arXiv PDF](https://arxiv.org/pdf/1808.01400).
- Representation: "represents a code snippet as the set of compositional paths in its abstract syntax tree (AST) and uses attention to select the relevant paths while decoding"; "significantly outperforms previous models that were specifically designed for programming languages, as well as state-of-the-art NMT models" (summarisation/captioning tasks, not VD). - [arXiv abs](https://arxiv.org/abs/1808.01400)

**17. ASTNN, ICSE 2019 - VERIFIED (author-hosted PDF)**
- Bib: Jian Zhang, Xu Wang, Hongyu Zhang, Hailong Sun, Kaixuan Wang, Xudong Liu. "A Novel Neural Source Code Representation Based on Abstract Syntax Tree." *2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE)*, pp. 783-794. DOI 10.1109/ICSE.2019.00086.
- Read: [author PDF](http://hongyujohn.github.io/ASTNN.pdf), [Crossref](https://api.crossref.org/works/10.1109/ICSE.2019.00086).
- Representation: "ASTNN splits each large AST into a sequence of small statement trees, and encodes the statement trees to vectors... Based on the sequence of statement vectors, a bidirectional RNN model is used". Motivation relevant to long functions: "the maximal node number/depth of ASTs of common code fragments in C and Java are 7,027/76 and 15,217/192". Evaluated on "source code classification and code clone detection" (not VD). - [author PDF](http://hongyujohn.github.io/ASTNN.pdf)

**18. TBCNN, AAAI 2016 - VERIFIED (arXiv abstract/full text; pages not returned by Crossref)**
- Bib: Lili Mou, Ge Li, Lu Zhang, Tao Wang, Zhi Jin. "Convolutional Neural Networks over Tree Structures for Programming Language Processing." *Proceedings of the AAAI Conference on Artificial Intelligence* 30(1), 2016. DOI 10.1609/aaai.v30i1.10139. arXiv:1409.5718.
- Read: [arXiv abs](https://arxiv.org/abs/1409.5718), [Crossref](https://api.crossref.org/works/10.1609/aaai.v30i1.10139).
- Representation: "a convolution kernel is designed over programs' abstract syntax trees to capture structural information"; tasks: "classifying programs according to functionality, and detecting code snippets of certain patterns". - [arXiv abs](https://arxiv.org/abs/1409.5718)

**AST models inside VD comparisons (VERIFIED):**
- Siow et al. 2022 (item 19), Devign, accuracy / F1: Code2Vec 0.7180 / 0.7192; Tree-LSTM 0.7100 / 0.7209; BiLSTM 0.7131 / 0.7162; GGNN 0.7158 / 0.7344. "An interesting finding is that Code2Vec performs better than BiLSTM in both code classification and vulnerability detection". GGNN restricted to AST edges: 0.7033 / 0.7125. - [arXiv PDF](https://arxiv.org/pdf/2203.11790)
- Steenhoek et al. 2023 (item 20): "Code2Vec used multilayer perceptron (MLP) on AST"; "Code2Vec reported the least variability compared to the GNN and transformer models." - [arXiv PDF](https://arxiv.org/pdf/2212.08109)

### Inferences
- AST-path/tree models are a middle ground (structure without full dependence graphs); in the only VD head-to-heads found they are within about 0.015 F1 of GGNN on Devign, which is small relative to the seed variability Steenhoek et al. report (Q5).

### Gaps
- No primary source read here evaluates code2seq, ASTNN or TBCNN on a VD benchmark against code PLMs; community replications (e.g., a GitHub replication of ASTNN for VD) were not treated as citable.

## Q4. Image representation: VulCNN and its efficiency argument

### Takeaway
VulCNN (ICSE 2022) turns each function's **PDG** into a 3-channel image (degree, Katz and closeness centrality over sent2vec line embeddings) and classifies it with a CNN; it reports being about 6x faster than Devign and about 4x faster than VulDeePecker/SySeVR, but it is graph-derived, is slower than a pure token CNN, and PDG extraction still takes more than 87% of its runtime.

### Cited Findings

**19a. VulCNN, ICSE 2022 - VERIFIED (author-hosted full PDF), except accuracy values (in a bar chart only)**
- Bib: Yueming Wu, Deqing Zou, Shihan Dou, Wei Yang, Duo Xu, Hai Jin. "VulCNN: An Image-inspired Scalable Vulnerability Detection System." *Proceedings of the 44th International Conference on Software Engineering (ICSE '22)*, Pittsburgh, May 21-29 2022, pp. 2365-2376. DOI 10.1145/3510003.3510229.
- Read: [author PDF](https://wu-yueming.github.io/Files/ICSE2022_VulCNN.pdf), [Crossref](https://api.crossref.org/works/10.1145/3510003.3510229).
- Motivation: "text-based techniques are scalable but not accurate due to the lack of program semantics. Graph-based methods are accurate but not scalable since graph analysis is typically time-consuming." - [author PDF](https://wu-yueming.github.io/Files/ICSE2022_VulCNN.pdf)
- Representation: "we first conduct program analysis to distill the program semantics of a function into a program dependency graph (PDG)... we treat it as a social network and apply centrality analysis"; "we leverage three different centralities (i.e., degree centrality [25], katz centrality [31], and closeness centrality [25])"; "we select 100 lines of code as our final threshold to generate the input images". - [author PDF](https://wu-yueming.github.io/Files/ICSE2022_VulCNN.pdf)
- Efficiency numbers: "VulCNN is about four times faster than VulDeePecker and SySeVR, about 15 times faster than VulDeeLocator, and about six times faster than Devign." "As for TokenCNN... it is the fastest. On average, it only takes 0.25 seconds to finish the analysis of a function"; "although VulCNN is not as scalable as TokenCNN". Image step: "More than 98% PDGs are able to be converted into images in one second, and the average runtime overhead [of this] phase is 0.26 seconds." Bottleneck: "the most time-consuming phase is to extract the PDG of functions, this phase occupies more than 87% of the total processing runtime." Case study: 600,233 functions, more than 25 million lines of code, images generated "in about 253 minutes" after PDGs; 73 vulnerabilities not reported in NVD (47 in older versions, 26 in latest versions). - [author PDF](https://wu-yueming.github.io/Files/ICSE2022_VulCNN.pdf)
- Dataset: "13,687 vulnerable functions and 26,970 non-vulnerable functions"; claims "better accuracy than eight state-of-the-art vulnerability detectors (i.e., Checkmarx, FlawFinder, RATS, TokenCNN, VulDeePecker, SySeVR, VulDeeLocator, and Devign)". - [author PDF](https://wu-yueming.github.io/Files/ICSE2022_VulCNN.pdf)
- UNVERIFIED: a search-engine summary states VulCNN reaches "an accuracy of 82%" and "TPR of 94%"; in the PDF these are only drawn in Figure 9 (bar chart) and could not be read as text.

### Inferences
- VulCNN's speedup is a property of the classifier (CNN on a fixed-size image vs GNN message passing), not of avoiding program analysis; for a pipeline where graph extraction is the cost, it would save little.

### Gaps
- Exact per-tool accuracy/TPR/TNR values for VulCNN's comparison (Figure 9) are not available as text.

## Q5. Comparative studies and counter-evidence (graph vs other representations)

### Takeaway
Evidence that graphs beat other representations for VD comes mainly from pre-PLM comparisons with small margins (Siow et al.: GGNN F1 0.7344 vs BiLSTM 0.7162 on one Devign split, and code2vec/GAT beat GGNN on accuracy). Once pre-trained code LMs and enough data are used, LMs beat the GNN ReVeal (DiverseVul: 47.15 vs 29.76 F1), but on small data they tie (CVEFixes: ReVeal 12.8 vs LMs 8.5-16.3). Graph and transformer VD models disagree heavily on individual predictions (only 7% of Devign test inputs agreed by all 9 models). The strongest counter-evidence is representation-agnostic: transformer VD models cannot tell a vulnerable function from its patch (accuracy 0.294-0.527 on VulnPatchPairs; PrimeVul pair-wise correct 1.06-3.01%).

### Cited Findings

**19. Siow et al., SANER 2022 - VERIFIED**
- Bib: Jing Kai Siow, Shangqing Liu, Xiaofei Xie, Guozhu Meng, Yang Liu. "Learning Program Semantics with Code Representations: An Empirical Study." *2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)*, pp. 554-565. DOI 10.1109/SANER53432.2022.00073. arXiv:2203.11790.
- Read: [arXiv abs](https://arxiv.org/abs/2203.11790), [arXiv PDF](https://arxiv.org/pdf/2203.11790), [Crossref](https://api.crossref.org/works/10.1109/SANER53432.2022.00073).
- Setup: four families (feature, sequence, tree, graph); VD on Devign, deduplicated, 48,158 functions after Joern, 80/10/10 split. No pre-trained transformer is included; the "Transformer Encoder" is trained from scratch.
- Headline: "(1) The graph-based representation is superior to the other selected techniques across these tasks." - [arXiv abs](https://arxiv.org/abs/2203.11790)
- Table II, VD accuracy / F1: SVM 0.5223 / 0.5144; Naive Bayes 0.6934 / 0.6762; XGBoost 0.7056 / 0.6951; LSTM 0.7098 / 0.7135; BiLSTM 0.7131 / 0.7162; Transformer Encoder 0.4796 / 0.6482; Code2Vec 0.7180 / 0.7192; Tree-LSTM 0.7100 / 0.7209; GCN 0.7015 / 0.7289; GAT 0.7278 / 0.7306; GGNN 0.7158 / 0.7344. Table II gives single values with no variance. - [arXiv PDF](https://arxiv.org/pdf/2203.11790)
- Table IV (GGNN by graph type), VD accuracy / F1: AST 0.7033 / 0.7125; CFG 0.7042 / 0.7085; CDG 0.7160 / 0.7120; DDG 0.7235 / 0.7133; CPG 0.7158 / 0.7344. "on the vulnerability detection, DDG can achieve higher performance". - [arXiv PDF](https://arxiv.org/pdf/2203.11790)
- Internal inconsistency: the body says GGNN beats non-graph representations by "1.35%-22.0% in F1-Score of vulnerability detection, and 3.12-30.37% in the accuracy of clone detection", but the RQ1 answer box says "30.37% F1-Score in vulnerability detection". Also on node embeddings: "on the vulnerability detection, using type embedding has a negative impact." - [arXiv PDF](https://arxiv.org/pdf/2203.11790)

**19b. Bagheri & Hegedűs, QUATIC 2021 - VERIFIED**
- Bib: Amirreza Bagheri, Péter Hegedűs. "A Comparison of Different Source Code Representation Methods for Vulnerability Prediction in Python." In *Quality of Information and Communications Technology (QUATIC 2021)*, Communications in Computer and Information Science, Springer, 2021, pp. 267-281. DOI 10.1007/978-3-030-85347-1_20. arXiv:2108.02044.
- Read: [arXiv abs](https://arxiv.org/abs/2108.02044), [arXiv PDF](https://arxiv.org/pdf/2108.02044), [Crossref](https://api.crossref.org/works/10.1007/978-3-030-85347-1_20).
- Scope: compares **text embeddings only** (word2vec, fastText, BERT) feeding the same LSTM, on Python vulnerability-fix data mined from GitHub; no graph or AST representation is tested.
- Numbers: "LSTM models using the BERT-based code representation achieve, on average, an accuracy of 93.8%, a recall of 83.2%, a precision of 91.4%, and an an F1 score of 87.1%. The models based on the word2vec code representation achieve, on average, an accuracy of 91%, a recall of 86.1%, a precision of 88.2%, and an F1 score of 85.6%. While fastText code representation based models achieve, on average, an accuracy of 91.8%, a recall of 86.4%, a precision of 85.1%, and an F1 score of 84%". RQ1: "We did not observe significant differences in the vulnerability prediction performances of the LSTM models trained on different code embeddings." (The same paragraph contains a fourth, unattributed set "accuracy of 91%, a recall of 81%, a precision of 90% and an F1 score of 82%", apparently an editing error.) - [arXiv PDF](https://arxiv.org/pdf/2108.02044)

**20. Steenhoek et al., ICSE 2023 - VERIFIED**
- Bib: Benjamin Steenhoek, Md Mahbubur Rahman, Richard Jiles, Wei Le. "An Empirical Study of Deep Learning Models for Vulnerability Detection." *2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)*, pp. 2237-2248. DOI 10.1109/ICSE48619.2023.00188. arXiv:2212.08109. Artifact DOI 10.6084/m9.figshare.20791240.
- Read: [arXiv abs](https://arxiv.org/abs/2212.08109), [arXiv PDF](https://arxiv.org/pdf/2212.08109), [Crossref](https://api.crossref.org/works/10.1109/ICSE48619.2023.00188).
- Models: 11 reproduced; "Devign [47] and ReVeal [7] used GNN on property graphs... ReGVD [31] used GNN on tokens. Code2Vec used multilayer perceptron (MLP) on AST. VulDeeLocator [22] and SySeVR [23] are based the sequence models of RNN and Bi-LSTMs. Recent deep learning detection used pre-trained transformers, including CodeBERT [14], VulBERTa-CNN [16], VulBERTa-MLP, PLBART [2] and LineVul [15]"; 9 models run on Devign and MSR (Big-Vul).
- Seed variability: "on average 34.9% test data (30.6% total data) reported different predictions dependent on the seeds used in training. The GNN models that work on property graph ranked the top 2 variability; especially for ReVeal, for 50% of the test data, its outputs changed between runs." (3 seeds, Devign.) - [arXiv PDF](https://arxiv.org/pdf/2212.08109)
- Agreement (graph vs transformer): "only 7% of the test data (and 7% total data) are agreed by all the models. The 3 GNN models agreed on 20% of test examples (and 25% total), whereas the 3 top performing transformers (LineVul, PLBART, and VulBERTa-CNN) agreed on 34% test data (and 44% total). But when we compared all 5 transformer models, only 22% of test examples (and 29% total) are agreed." - [arXiv PDF](https://arxiv.org/pdf/2212.08109)
- What they learned: models highlight `for`, `if`, `while`, signatures, `alloc`/`memset`/`memcpy` and error-printing lines; "the probability of error occurring in the important feature sets is 2.79 times of the probability of error occurring in the program on average". Feature overlap: "Linevul and ReGVD have a maximum overlap among all the model pairs. Among the 10 lines ranked as the important features, the two models shared an average of 6.88 lines"; "Devign, as the only GNN model based on the property graph, has a low overlap with the other models and the lowest is with PLBART, on average 3.38 lines." - [arXiv PDF](https://arxiv.org/pdf/2212.08109)
- Data: "all 9 models performed better on the easy dataset than on the difficult set. The average difference between easy/difficult performance was 10.3%"; "Comparing 100% data with 10%, the F1 on test set reported no difference when we take an average over all the models for the Balanced dataset." - [arXiv PDF](https://arxiv.org/pdf/2212.08109)

**21. PrimeVul (Ding et al.), ICSE 2025 - VERIFIED**
- Bib: Yangruibo Ding, Yanjun Fu, Omniyyah Ibrahim, Chawin Sitawarin, Xinyun Chen, Basel Alomair, David Wagner, Baishakhi Ray, Yizheng Chen. "Vulnerability Detection with Code Language Models: How Far Are We?" *2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)*, pp. 1729-1741. DOI 10.1109/ICSE55347.2025.00038. arXiv:2403.18624.
- Read: [arXiv abs](https://arxiv.org/abs/2403.18624), [arXiv PDF](https://arxiv.org/pdf/2403.18624), [Crossref](https://api.crossref.org/works/10.1109/ICSE55347.2025.00038).
- Headline: "a state-of-the-art 7B model scored 68.26% F1 on BigVul but only 3.09% F1 on PrimeVul." - [arXiv abs](https://arxiv.org/abs/2403.18624)
- Table V (train -> test; Acc / F1 / VD-S (lower better) / P-C pair-wise correct): CodeT5 60M BV->BV 95.67 / 64.93 / 77.30 / 24.98, PV->PV 96.67 / 19.7 / 89.93 / 1.06; CodeBERT 125M BV->BV 95.57 / 62.88 / 81.77 / 22.60, PV->PV 96.87 / 20.86 / 88.78 / 1.77; UniXcoder 125M BV->BV 96.46 / 65.46 / 62.30 / 39.60, PV->PV 96.86 / 21.43 / 89.21 / 1.60; StarCoder2 7B BV->BV 96.20 / 68.26 / 69.14 / 35.23, PV->PV 97.02 / 18.05 / 89.64 / 2.30; CodeGen2.5 7B BV->BV 96.57 / 67.30 / 61.73 / 40.84, PV->PV 96.65 / 19.61 / 91.51 / 3.01. VD-S = FNR at FPR <= 0.5%. - [arXiv PDF](https://arxiv.org/pdf/2403.18624)
- Interpretation by the authors: "these models make decisions primarily based on textual similarity, without considering the underlying root causes or fixes of the vulnerabilities." - [arXiv PDF](https://arxiv.org/pdf/2403.18624)
- Scope note: PrimeVul evaluates only code LMs; its full text contains no GNN/graph baseline (full-text search), so it is not a graph-vs-sequence comparison.

**22. DiverseVul (Chen et al.), RAID 2023 - VERIFIED**
- Bib: Yizheng Chen, Zhoujie Ding, Lamya Alowain, Xinyun Chen, David Wagner. "DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection." *Proceedings of the 26th International Symposium on Research in Attacks, Intrusions and Defenses (RAID '23)*, Hong Kong, Oct 16-18 2023, pp. 654-668. DOI 10.1145/3607199.3607242. arXiv:2304.00409.
- Read: [arXiv abs](https://arxiv.org/abs/2304.00409), [arXiv PDF](https://arxiv.org/pdf/2304.00409), [Crossref](https://api.crossref.org/works/10.1145/3607199.3607242).
- Abstract claim: LLMs are "outperforming Graph Neural Networks (GNNs) with code-structure features in our experiments." GNN = ReVeal (GGNN over code property graphs, 1.28M parameters) vs 10 LMs (RoBERTa, CodeBERT, GraphCodeBERT, GPT-2 Base, CodeGPT, PolyCoder, T5 Base, CodeT5 Small, CodeT5 Base, NatGen; 60M-220M). - [arXiv PDF](https://arxiv.org/pdf/2304.00409)
- Numbers: "When trained on all available data (Previous + DiverseVul), LLMs perform significantly better than the ReVeal model: the ReVeal model achieves a 29.76 F1 score, while LLMs achieve F1 scores from 31.96 to 47.15." On small data: "on CVEFixes, the largest previously released dataset, the ReVeal model (a GNN) achieves 12.8 F1 score, vs F1 scores of 8.5-16.3 for LLMs"; "ReVeal is even better than 6 LLMs (out of 10) in this setting." "larger datasets improve the performance of ReVeal only modestly, but improve the performance of LLMs significantly." - [arXiv PDF](https://arxiv.org/pdf/2304.00409)
- Authors' own caveat: "Comparing between ReVeal and LLMs is arguably unfair since ReVeal has 1-2 orders of magnitude fewer parameters than LLMs. We do not know whether a larger GNN could be competitive with LLMs." - [arXiv PDF](https://arxiv.org/pdf/2304.00409)
- Other relevant findings: generalisation "from a F1 score of 49% on seen projects to only 9.4% on unseen projects"; "pretraining models on code using MLM or next token prediction techniques does not yield significant improvements in detecting C/C++ vulnerabilities. While CodeBERT, GraphCodeBERT, and CodeGPT have not pretrained on C/C++, PolyCoder has pretrained over C/C++ code for next token prediction, which still does not help detecting C/C++ vulnerabilities." - [arXiv PDF](https://arxiv.org/pdf/2304.00409)

**23. Risse & Böhme, USENIX Security 2024 - VERIFIED (arXiv full text; USENIX page numbers not checked)**
- Bib: Niklas Risse, Marcel Böhme. "Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection." *Proceedings of the 33rd USENIX Security Symposium (USENIX Security 2024)*, Philadelphia, PA, Aug 2024 (arXiv journal-ref). arXiv:2306.17193.
- Read: [arXiv abs](https://arxiv.org/abs/2306.17193), [arXiv PDF](https://arxiv.org/pdf/2306.17193).
- Models: "UniXcoder [13], CoTexT [28], VulBERTa [15], PLBart [1], and CodeBERT [12]" plus GraphCodeBERT, i.e., six transformer models and **no GNN**; datasets CodeXGLUE/Devign and VulDeePecker. - [arXiv PDF](https://arxiv.org/pdf/2306.17193)
- Findings: "(a) that state-of-the-art models severely overfit to unrelated features for predicting the vulnerabilities in the testing data, (b) that the performance gained by data augmentation does not generalize beyond the specific augmentations applied during training, and (c) that state-of-the-art ML4VD techniques are unable to distinguish vulnerable functions from their patches." - [arXiv abs](https://arxiv.org/abs/2306.17193)
- Numbers: on VulnPatchPairs, "the accuracy drops dramatically (between 0.294 and 0.527). Even the best model (VulBERTa) is only 0.027 points better than a random guesser. On average, the accuracy is worse than random guessing." Training on VulnPatchPairs and testing on standard data gives "between 0.546 and 0.575". Most harmful semantic-preserving transformations: "transformations that insert statements (e.g. t4, t5, t8, and t10) or reorder statements (e.g. t2 and t6) seem to have a higher impact than the other types." - [arXiv PDF](https://arxiv.org/pdf/2306.17193)

**24. Pre-trained LMs vs GNN baselines inside a hybrid paper (Vul-LMGNNs, Table 2/3) - VERIFIED** (full bib in Q6 item 27)
- F1 (%) of plain CodeBERT vs graph baselines: DiverseVul CodeBERT 23.44 vs ReVeal 12.42, ReGVD 13.73, Devign(AST) 9.28; Draper VDISC 82.39 vs ReVeal 78.01, ReGVD 79.66, Devign(AST) 63.49; Devign dataset 59.53 vs ReVeal 56.59, ReGVD 53.75, Devign(AST) 56.60; ReVeal dataset 38.19 vs ReVeal 33.87, ReGVD 23.65, Devign(AST) 33.91. - [arXiv PDF](https://arxiv.org/pdf/2404.14719)

### Inferences
- Siow et al.'s "graph is superior" conclusion rests on a single Devign split without variance, and the VD margin (GGNN minus BiLSTM F1 = +0.018; code2vec and GAT ahead of GGNN on accuracy) is small next to Steenhoek et al.'s finding that 34.9% of Devign test predictions flip with the seed. It should be cited as "graphs slightly ahead of from-scratch sequence models", not as evidence against pre-trained sequence encoders, which it did not test.
- DiverseVul and the Vul-LMGNNs baselines agree that a fine-tuned code LM beats GNN-only VD models once data is large enough; DiverseVul also shows the ordering can reverse on small data (CVEFixes), which matters for small target sets.
- The vulnerable-vs-patch failures (Risse & Böhme; PrimeVul P-C at most 3.01%) were measured on transformer models only, so they do not show that graphs would do better; they show that neither family has been shown to learn the vulnerability itself rather than textual correlates.
- Steenhoek et al.'s agreement numbers (7% all-model agreement; Devign sharing on average 3.38 of 10 important lines with PLBART) are the strongest verified evidence that graph and sequence models learn different signals, which is the usual argument for fusion.

### Gaps
- No verified head-to-head found where a GNN of comparable parameter count to a code PLM is trained on the same large dataset (DiverseVul explicitly leaves this open).
- None of the comparative studies read here uses Python VD data except Bagheri & Hegedűs, which compares only text embeddings.

## Q6. Hybrid graph + sequence models and their measured gains (GREAT, VELVET, Vul-LMGNNs)

### Takeaway
Hybrids report gains, but they are task- and backbone-dependent: GREAT (not VD) reports +10-15% over a strong GGNN on variable misuse; VELVET's graph+transformer ensemble mainly helps statement localisation (D2A top-1 43.6% vs 39.8% transformer-only) while its function-level F1 gain over the transformer alone is 0.2 points; Vul-LMGNNs' gain over the same LM alone ranges from +0.10 to +13.03 F1 depending on dataset and backbone.

### Cited Findings

**25. GREAT / Graph Sandwiches (Hellendoorn et al.), ICLR 2020 - PARTIAL (abstract only; OpenReview full text blocked by a bot challenge)**
- Bib: Vincent J. Hellendoorn, Charles Sutton, Rishabh Singh, Petros Maniatis, David Bieber. "Global Relational Models of Source Code." ICLR 2020. OpenReview id B1lnbRNtwr (no DOI).
- Read: [ICLR 2020 virtual page](https://iclr.cc/virtual_2020/poster_B1lnbRNtwr.html) (author order taken from here), [Google Research page](https://research.google/pubs/global-relational-models-of-source-code/) (lists the same five authors in a different order).
- Representation: "Graph Sandwiches, which wrap traditional (gated) graph message-passing layers in sequential message-passing layers; and Graph Relational Embedding Attention Transformers (GREAT for short), which bias traditional Transformers with relational information from graph edge types."
- Finding (variable-misuse task, not VD): "Starting with a graph-based model that already improves upon the prior state-of-the-art for this task by 20%, we show that our proposed hybrid models improve an additional 10-15%, while training both faster and using fewer parameters." - [ICLR virtual page](https://iclr.cc/virtual_2020/poster_B1lnbRNtwr.html)

**26. VELVET (Ding et al.), SANER 2022 - VERIFIED**
- Bib: Yangruibo Ding, Sahil Suneja, Yunhui Zheng, Jim Laredo, Alessandro Morari, Gail Kaiser, Baishakhi Ray. "VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements." *2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)*, pp. 959-970. DOI 10.1109/SANER53432.2022.00114. arXiv:2112.10893.
- Read: [arXiv abs](https://arxiv.org/abs/2112.10893), [arXiv PDF](https://arxiv.org/pdf/2112.10893), [Crossref](https://api.crossref.org/works/10.1109/SANER53432.2022.00114).
- Important scope note: its "sequence" branch is a transformer over **graph node embeddings**, not a token-sequence PLM: "Note that the transformer here is a bit different than the commonly used transformer of CodeBert [35], Roberta [50] etc., where the inputs are token sequences. In our case, the input consists of pre-processed graph nodes". - [arXiv PDF](https://arxiv.org/pdf/2112.10893)
- Table III (localisation, vulnerable functions only), top-1: Juliet Ensemble 99.6%, GGNN 98.1%, Transformer 99.4%; D2A Ensemble 43.6%, GGNN 33.8%, Transformer 39.8% (top-3 D2A 63.9 / 54.9 / 62.4%). "VELVET-ENSEMBLE improves the top-1 accuracy of single VELVET-GGNN by around 29.0% and VELVET-TRANSFORMER by 9.5%." - [arXiv PDF](https://arxiv.org/pdf/2112.10893)
- Table II (joint training, D2A), Vul-CLS F1: Ensemble 59.0%, GGNN 51.0%, Transformer 58.8%; Vul-LOC acc 30.1 / 19.6 / 29.3%. - [arXiv PDF](https://arxiv.org/pdf/2112.10893)
- Complementarity: on 133 recent D2A-TEST vulnerabilities "two models locate 37 vulnerabilities in common; besides these, GGNN can further locate 8 individual vulnerabilities and Transformer alone can locate 16"; GGNN-only correct functions average "27.75 lines and 138.0 graph nodes", Transformer-only "48.38 lines and 288.6 graph nodes". - [arXiv PDF](https://arxiv.org/pdf/2112.10893)

**27. Vul-LMGNNs (Liu et al.), Information Fusion 2025 - VERIFIED (arXiv full text)**
- Bib: Ruitong Liu, Yanbin Wang, Haitao Xu, Jianguo Sun, Fan Zhang, Peiyue Li, Zhenhao Guo. "Vul-LMGNNs: Fusing language models and online-distilled graph neural networks for code vulnerability detection." *Information Fusion* 115 (2025) 102748. DOI 10.1016/j.inffus.2024.102748. arXiv:2404.14719 (an earlier arXiv version appears in search listings as "Source Code Vulnerability Detection: Combining Code Language Models and Code Property Graphs").
- Read: [arXiv abs](https://arxiv.org/abs/2404.14719), [arXiv PDF](https://arxiv.org/pdf/2404.14719), [Crossref](https://api.crossref.org/works/10.1016/j.inffus.2024.102748).
- Representation: code property graph + gated GNN whose node embeddings are initialised by a code LM, online knowledge distillation, and "late fusion via linear interpolation" of LM and GNN predictions. Claim: "outperform 17 state-of-the-art approaches". - [arXiv abs](https://arxiv.org/abs/2404.14719)
- Gain over the same LM alone, F1 (%), from Tables 2-3 (LM alone -> Vul-LMGNNs with that LM; differences are my arithmetic):
  - DiverseVul: CodeBERT 23.44 -> 23.54 (+0.10); GraphCodeBERT 21.40 -> 23.90 (+2.50); CodeT5-Small 24.83 -> 30.43 (+5.60); CodeT5-Base 26.40 -> 32.59 (+6.19).
  - Draper VDISC: 82.39 -> 83.87 (+1.48); 83.95 -> 84.11 (+0.16); 84.51 -> 85.48 (+0.97); 85.12 -> 85.83 (+0.71).
  - Devign: 59.53 -> 60.16 (+0.63); 58.96 -> 61.01 (+2.05); 59.53 -> 63.82 (+4.29); 60.48 -> 63.44 (+2.96).
  - ReVeal: 38.19 -> 51.22 (+13.03); 41.74 -> 48.57 (+6.83); 43.99 -> 50.58 (+6.59); 44.56 -> 53.09 (+8.53). - [arXiv PDF](https://arxiv.org/pdf/2404.14719)
- Interpolation ablation (partial DiverseVul): "Accuracy improves consistently as λ increases, reaching its peak at λ=0.8, slightly outperforming the GGNN or CodeBert alone (at λ = 0 or 1). The achieved accuracy is 90.24%." - [arXiv PDF](https://arxiv.org/pdf/2404.14719)
- Data-quality flags (my reading of the tables): VulDeePecker and ReVeal have identical rows on Draper VDISC (78.15 / 78.50 / 77.53 / 78.01), and Devign(AST) and CFExplainer have identical rows on the Devign dataset (57.66 / 56.96 / 56.25 / 56.60); the tables report single values without variance or seed count. - [arXiv PDF](https://arxiv.org/pdf/2404.14719)

### Inferences
- In VD, the verified benefit of adding a graph branch to a sequence model is clearest for **localisation** and for **long functions** (VELVET) and for the weaker/encoder-decoder backbones in Vul-LMGNNs; with CodeBERT as the backbone the DiverseVul gain is +0.10 F1, i.e., nil.
- Several of the Vul-LMGNNs gains (e.g., Draper +0.16 to +1.48 F1) are single-run differences without variance, the same weakness flagged above for Siow et al.

### Gaps
- GREAT's full text (numbers per model on VarMisuse) could not be read; it is not a VD paper.
- Other 2023-2026 LM+graph VD hybrids (e.g., structure-aware pre-training such as PDBERT, which RevisitVD evaluates) were not read in primary form here; RevisitVD's abstract-level claim that syntax/semantics-aware pre-training helps (item 13) is the only verified statement on that line.
