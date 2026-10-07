# Why a GNN branch on top of a fine-tuned code transformer (CodeBERT) adds little or nothing for vulnerability detection (VD): evidence for explaining a measured null result

Status legend: **VERIFIED** = number/quote read in the primary full text (arXiv or author PDF converted with pdftotext, or the project's earlier verified notes, which are marked "reused"); **PARTIAL** = abstract/metadata only; **UNVERIFIED** = secondary source only. Bibliographic fields were checked against Crossref (`api.crossref.org`) or the arXiv/PMLR record unless stated. Quotes are verbatim except that PDF line-break hyphenation is joined and long dashes are written as "-". "Reused" items come from `research_notes/Trích dẫn dữ liệu và biểu diễn mã/graph_representations.md` and `other_representations_comparisons.md` (verified 2026-09-28) and were not re-verified. Everything else was read on 2026-10-05.

Measured pattern being explained (from the assignment, internal, not a publication): CodeBERT multi-window text branch + BABEL-style line graph (nodes = lines, node feature = mean of contextual CodeBERT token states of the line; 4 heuristic relations; 2 R-GCN layers + LSTM + max-pool; ~9.45M random-init parameters at lr 2e-5), concatenated with the text vector. Graph minus no-graph is within about +/-0.016 ROC and flips sign; a shuffled graph is at least as good as the real one (n=5); 63% of same-line-count SVEN Python vulnerable/patched pairs have byte-identical graphs in all 4 relations; 66% of indentation edges and 46% of co-use edges join adjacent lines; 75% of functions fit in one 510-token window.

Working hypotheses used in the assessments below:
- **H1 Redundancy**: self-attention over the whole function (one window for 75% of functions) already encodes the syntax/data-flow relations the heuristic edges add, and the node features are themselves contextual states.
- **H2 Non-discriminative structure**: vulnerable and patched versions differ by a few tokens inside a line, so the line graph is identical for most pairs and cannot separate them.
- **H3 Structure unused / overfit**: GNNs often do not exploit the given graph, so a shuffled graph performs the same.
- **H4 Noisy heuristic edges**: regex/indentation edges mis-model Python multi-line statements; structure only helps when it is accurate.
- **H5 Optimisation**: a large random-init branch trained at the backbone lr (2e-5) on ~456 training functions barely leaves initialisation.
- **H6 Smoothing/aggregation**: mean-pooled node features and message passing smooth away the token-level difference.

## Q1. Do GNN branches on top of pretrained transformers add more than marginal gains for code tasks and VD?

### Takeaway
The published "graph on PLM" gains for VD are small, inconsistent across datasets/backbones, or come from settings unlike ours. ReGVD's +1.39 over GraphCodeBERT is a GNN that *replaces* the transformer layers (it uses only CodeBERT's token-embedding layer); VELVET's +0.2 F1 is over a transformer on graph nodes, not a PLM; Vul-LMGNNs gives +0.10 F1 with CodeBERT on DiverseVul but up to +13.03 on the ReVeal set, single runs without variance; LineVD's gain (0.296 to 0.360 statement-level F1) is over a *frozen* CodeBERT, and "other graph-based combinations" were "comparable to using the model without a GNN". The clearest positive results (DeepDFA +1.35 to +3.55 F1, Sachan et al. +3 F1 with gold trees, UniXcoder AST +1.78 MAP@R) use accurate parser-built structure, purpose-built node features, larger data, or structure in pre-training rather than a randomly initialised add-on at fine-tuning time. The closest NLP analogue (Sachan et al., EACL 2021: a GNN stacked on BERT's output) found gains "highly contingent on the availability of human-annotated dependency parses" and "little to no gains" with predicted parses.

### Cited Findings

**[A1] ReGVD (ICSE 2022 Companion). VERIFIED (reused numbers; new read of the method)**
- Bib: Van-Anh Nguyen, Dai Quoc Nguyen, Van Nguyen, Trung Le, Quan Hung Tran, Dinh Phung. "ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection." ICSE '22 Companion, pp. 178-182, 2022. DOI 10.1145/3510454.3516865. arXiv:2110.07317. URL read: https://arxiv.org/pdf/2110.07317 - [arXiv PDF](https://arxiv.org/pdf/2110.07317)
- Number: "ReGVD produces the highest accuracy of 63.69%, gaining absolute improvements of 1.61% and 1.39% over CodeBERT and GraphCodeBERT" (CodeXGLUE/Devign set). - [arXiv PDF](https://arxiv.org/pdf/2110.07317)
- Architecture (decisive for how to cite it): "node features are initialized by only the token embedding layer of a pre-trained programming language (PL) model"; "To attain the advantage of the pre-trained PL model and also to make a fair comparison, we use only the token embedding layer of the pre-trained PL model to initialize node feature vectors for reporting our final results." - [arXiv PDF](https://arxiv.org/pdf/2110.07317)
- Optimiser: "the Adam initial learning rate ("lr") in 1e-4, 5e-4, 1e-3"; result is "the best model checkpoint, which obtains the highest accuracy on the validation set" (single number, no variance). - [arXiv PDF](https://arxiv.org/pdf/2110.07317)
- Context: PrimeVul measured CodeXGLUE/Devign vulnerable-label accuracy at 24% (reused, `label_quality_duplication.md`). - [PrimeVul arXiv PDF](https://arxiv.org/pdf/2403.18624)

**[A2] Vul-LMGNNs (Information Fusion 2025). VERIFIED (reused numbers; new read of the training setup)**
- Bib: Ruitong Liu, Yanbin Wang, Haitao Xu, Jianguo Sun, Fan Zhang, Peiyue Li, Zhenhao Guo. "Vul-LMGNNs: Fusing language models and online-distilled graph neural networks for code vulnerability detection." *Information Fusion* 115 (2025) 102748. DOI 10.1016/j.inffus.2024.102748. arXiv:2404.14719. - [arXiv PDF](https://arxiv.org/pdf/2404.14719)
- Gain over the same LM alone, F1 (%), Tables 2-3 (differences computed in the earlier notes): DiverseVul CodeBERT 23.44 -> 23.54 (+0.10), GraphCodeBERT +2.50, CodeT5-Small +5.60, CodeT5-Base +6.19; Draper +0.16 to +1.48; Devign +0.63 to +4.29; ReVeal +6.59 to +13.03 (CodeBERT 38.19 -> 51.22). Single values, no variance or seed count. - [arXiv PDF](https://arxiv.org/pdf/2404.14719)
- Design: CPG + gated GNN with LM-initialised node embeddings, online distillation, "late fusion via linear interpolation" of LM and GNN predictions. Setup: "the learning rate and batch size were set to 1e-4 and 64, respectively. The training was conducted over 20 epochs"; "functions with a node size exceeding 500 in the CPG were excluded from our analysis." - [arXiv PDF](https://arxiv.org/pdf/2404.14719)

**[A3] VELVET (SANER 2022). VERIFIED (reused)**
- Bib: Yangruibo Ding, Sahil Suneja, Yunhui Zheng, Jim Laredo, Alessandro Morari, Gail Kaiser, Baishakhi Ray. "VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements." SANER 2022, pp. 959-970. DOI 10.1109/SANER53432.2022.00114. arXiv:2112.10893. - [arXiv PDF](https://arxiv.org/pdf/2112.10893)
- Function-level Vul-CLS F1 on D2A (Table II): Ensemble 59.0%, GGNN 51.0%, Transformer 58.8% (+0.2 over the transformer). - [arXiv PDF](https://arxiv.org/pdf/2112.10893)
- Scope: "the transformer here is a bit different than the commonly used transformer of CodeBert [35], Roberta [50] etc., where the inputs are token sequences. In our case, the input consists of pre-processed graph nodes". Learning rates: "The learning rate for pre-training is 10^-4, and for finetuning is 10^-5." - [arXiv PDF](https://arxiv.org/pdf/2112.10893)

**[A4] LineVD (MSR 2022). VERIFIED (reused quote plus new read)**
- Bib: David Hin, Andrey Kan, Huaming Chen, M. Ali Babar. "LineVD: Statement-level Vulnerability Detection using Graph Neural Networks." MSR '22, pp. 596-607. DOI 10.1145/3524842.3527949. arXiv:2203.05181. - [arXiv PDF](https://arxiv.org/pdf/2203.05181)
- Backbone frozen: "we use multiple hidden layers, and freeze all parameters of the CodeBERT model during training." - [arXiv PDF](https://arxiv.org/pdf/2203.05181)
- RQ3 (statement level): "When comparing this GNN feature extraction combination with the model variation without GNN, we achieve an increase of 24% in F1 score (p < 0.01) from 0.296 to 0.360 ... However, the performance of other graph-based combinations is generally comparable to using the model without a GNN. This suggests that the program graph type and GNN type non-trivially affects the performance of the model." Graphs come from Joern PDG/CDG. - [arXiv PDF](https://arxiv.org/pdf/2203.05181)

**[A5] DeepDFA (ICSE 2024). VERIFIED**
- Bib: Benjamin Steenhoek, Hongyang Gao, Wei Le. "Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection." ICSE '24, pp. 1-13. DOI 10.1145/3597503.3623345 (Crossref). arXiv:2212.08108. URL read: https://arxiv.org/pdf/2212.08108 - [arXiv PDF](https://arxiv.org/pdf/2212.08108)
- Combining its GNN embedding with transformers (Big-Vul, mean (std) F1): LineVul 93.23 (0.31) vs DeepDFA+LineVul 96.40 (0.13); UniXcoder 95.11 (0.21) vs DeepDFA+UniXcoder 96.46 (0.09); "an F1 score of 96.46 (1.35 improvement)". Mixed/cross-project: "improving on UniXcoder's mean F1 score by 3.55 and 1.35 points respectively". - [arXiv PDF](https://arxiv.org/pdf/2212.08108)
- Why its node features differ from text embeddings: "Devign used word embeddings to encode statements into vector representations based on their unstructured text content. Such an encoding, even propagated through data dependency edges, cannot directly capture the dataflow patterns." GNN learning rate 1e-3 (Table 2). - [arXiv PDF](https://arxiv.org/pdf/2212.08108)

**[A6] GraphCodeBERT data-flow ablation (ICLR 2021). VERIFIED (reused)**
- Bib: Daya Guo et al. "GraphCodeBERT: Pre-training Code Representations with Data Flow." ICLR 2021. arXiv:2009.08366. - [arXiv PDF](https://arxiv.org/pdf/2009.08366)
- "After ablating the data flow totally, we can see that the performance drops from 71.3% to 69.3%" (code search, overall MRR). The paper does not evaluate VD. Data flow enters through pre-training and graph-guided attention, not a separate GNN. - [arXiv PDF](https://arxiv.org/pdf/2009.08366)

**[A7] UniXcoder AST ablation (ACL 2022). VERIFIED**
- Bib: Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, Jian Yin. "UniXcoder: Unified Cross-Modal Pre-training for Code Representation." ACL 2022 (Long), pp. 7212-7225. DOI 10.18653/v1/2022.acl-long.499. arXiv:2203.03850. URL read: https://arxiv.org/pdf/2203.03850 - [arXiv PDF](https://arxiv.org/pdf/2203.03850)
- POJ-104 MAP@R: UniXcoder 90.52 vs "-w/o AST" 88.74 (Table 1, read from the table row order). Text: "injecting AST can boost the performance on all code understanding tasks. However, AST does not bring improvements on generation tasks, which may require a better way to incorporate AST for generation tasks." AST is a flattened token sequence in pre-training, not a GNN. No VD task. - [arXiv PDF](https://arxiv.org/pdf/2203.03850)

**[A8] Sachan et al., syntax GNN stacked on BERT (EACL 2021; NLP analogue). VERIFIED**
- Bib: Devendra Singh Sachan, Yuhao Zhang, Peng Qi, William Hamilton. "Do Syntax Trees Help Pre-trained Transformers Extract Information?" EACL 2021, pp. 2647-2661. DOI 10.18653/v1/2021.eacl-main.228. arXiv:2008.09084. URL read: https://arxiv.org/pdf/2008.09084 - [arXiv PDF](https://arxiv.org/pdf/2008.09084)
- Design matches ours ("Late Fusion"): "a late fusion approach, which applies a graph neural network on the output of a transformer"; "During the finetuning step, the new parameters in each model are randomly initialized while the existing parameters are initialized from pre-trained BERT." - [arXiv PDF](https://arxiv.org/pdf/2008.09084)
- Result: "we find that their performance gains are highly contingent on the availability of human-annotated dependency parses, which raises important questions regarding the viability of syntax-augmented transformers in real-world applications." With gold trees the best variant beats "a fine-tuned BERT model by over 3 F1 points" (SRL), but "using off-the-shelf parses from the Stanza toolkit (Qi et al., 2020) provides little to no gains in F1 scores ... This is mainly due to the low in-domain accuracy of the predicted parses" (Stanza UAS 84.2% on CoNLL-2012). - [arXiv PDF](https://arxiv.org/pdf/2008.09084)
- Optimiser: "We observed that the initial learning rate of 2e-5 with a linear decay worked well for all the tasks"; 10-20 epochs on CoNLL/TACRED. Results averaged "over five runs with different random seeds". - [arXiv PDF](https://arxiv.org/pdf/2008.09084)

**[A9] Ahmad et al., AST in a code Transformer (ACL 2020). VERIFIED**
- Bib: Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang. "A Transformer-based Approach for Source Code Summarization." ACL 2020, pp. 4998-5007. DOI 10.18653/v1/2020.acl-main.449. arXiv:2005.00653. - [arXiv PDF](https://arxiv.org/pdf/2005.00653)
- "Our experimental findings suggest that the incorporation of AST information in the Transformer does not result in an improvement in source code summarization. We hypothesize that the exploitation of the code structure information in summarization has limited advantage, and it diminishes as the Transformer learns it implicitly with relative position representation." - [arXiv PDF](https://arxiv.org/pdf/2005.00653)

**[A10] Chirkova & Troshin, syntax in Transformers for code (ESEC/FSE 2021). VERIFIED**
- Bib: Nadezhda Chirkova, Sergey Troshin. "Empirical Study of Transformers for Source Code." ESEC/FSE '21, pp. 703-715. DOI 10.1145/3468264.3468611. arXiv:2010.07987. - [arXiv PDF](https://arxiv.org/pdf/2010.07987)
- Conclusions: "sequential relative attention is a simple, fast and not considered as the baseline in previous works mechanism that performs best in 3 out of 4 tasks (in some cases, similarly to other slower mechanisms)"; "combining sequential relative attention with GGNN Sandwich in the variable misuse task ... may further improve quality"; "in function naming, Transformer mostly relies on a set of types and values used in the program, hardly utilizing syntactic structure." Models trained from scratch (not PLMs). - [arXiv PDF](https://arxiv.org/pdf/2010.07987)

**[A11] Code Transformer, structure vs context (ICLR 2021). VERIFIED (counter-evidence, from scratch)**
- Bib: Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, Stephan Günnemann. "Language-Agnostic Representation Learning of Source Code from Structure and Context." ICLR 2021. arXiv:2103.11318. - [arXiv PDF](https://arxiv.org/pdf/2103.11318)
- Java-small method naming with pointer network, F1: full model 52.22, "w/o structure" 50.34, "w/o context" 49.45 (Table 3). Abstract: "multilingual training only from Context does not lead to the same improvements, highlighting the benefits of combining Structure and Context". - [arXiv PDF](https://arxiv.org/pdf/2103.11318)

**[A12] DiverseVul (RAID 2023): LMs vs a GNN. VERIFIED (reused)**
- Bib: Yizheng Chen, Zhoujie Ding, Lamya Alowain, Xinyun Chen, David Wagner. RAID '23, pp. 654-668. DOI 10.1145/3607199.3607242. arXiv:2304.00409. - [arXiv PDF](https://arxiv.org/pdf/2304.00409)
- "the ReVeal model achieves a 29.76 F1 score, while LLMs achieve F1 scores from 31.96 to 47.15"; on CVEFixes "ReVeal ... 12.8 F1 score, vs F1 scores of 8.5-16.3 for LLMs". Authors' caveat: "ReVeal has 1-2 orders of magnitude fewer parameters than LLMs". - [arXiv PDF](https://arxiv.org/pdf/2304.00409)

**[A13] GREAT / Graph Sandwiches (ICLR 2020). PARTIAL (reused; full text blocked by OpenReview bot check again on 2026-10-05)**
- Bib: Vincent J. Hellendoorn, Charles Sutton, Rishabh Singh, Petros Maniatis, David Bieber. "Global Relational Models of Source Code." ICLR 2020, OpenReview B1lnbRNtwr. - [ICLR virtual page](https://iclr.cc/virtual_2020/poster_B1lnbRNtwr.html)
- Abstract: hybrids "bias traditional Transformers with relational information from graph edge types" and "improve an additional 10-15%" over a graph model on variable misuse (not VD; transformers not pre-trained). - [ICLR virtual page](https://iclr.cc/virtual_2020/poster_B1lnbRNtwr.html)

**[A14] AST-Enhanced or AST-Overloaded? (2025, clone detection). PARTIAL-plus (abstract read in PDF) - PREPRINT, venue not verified**
- Bib: Zixian Zhang, Takfarinas Saber. arXiv:2506.14470 (Jun 2025). - [arXiv PDF](https://arxiv.org/pdf/2506.14470)
- "FA-AST frequently introduces structural complexity that harms performance. Notably, GMN outperforms others even with standard AST representations, highlighting its superior cross-code similarity detection and reducing the need for enriched structures." - [arXiv PDF](https://arxiv.org/pdf/2506.14470)

**[A15] Concatenation hybrid vs matched-capacity MLP (Elliptic fraud detection, 2026). PARTIAL-plus (abstract read in PDF) - PREPRINT, single author, outside code**
- Bib: Saket Maganti. "When Graph Structure Becomes a Liability: A Critical Re-Evaluation of Graph Neural Networks for Bitcoin Fraud Detection under Temporal Distribution Shift." arXiv:2604.19514 (21 Apr 2026). - [arXiv PDF](https://arxiv.org/pdf/2604.19514)
- "the same hybrid falls to F1 = 0.699 ± 0.015, and the GNN contributes a statistically reliable but small +0.018 F1 lift over a matched-capacity MLP substitute (p = 0.015, d = +1.20)"; "A 10-seed edge-shuffle ablation further shows that on Elliptic randomly shuffled edges outperform the real transaction graph by 8.9 F1 points". - [arXiv PDF](https://arxiv.org/pdf/2604.19514)

### Inferences
- Assessment [A1] ReGVD: **does not** show a GNN adding to a full PLM. Its graph branch replaces CodeBERT's 12 transformer layers, so "+1.39 over GraphCodeBERT" compares GNN-on-embeddings with a full transformer on a dataset with 24% label accuracy, single run. It should not be cited as "GNN on top of GraphCodeBERT gives +1.39". Consistent with H1 only indirectly (a cheap token graph can roughly replace attention on this benchmark).
- Assessment [A2] Vul-LMGNNs: supports "gain depends on dataset and backbone", and the CodeBERT/DiverseVul +0.10 is the closest analogue to our null. It also uses parser CPGs, a distillation objective, prediction-level interpolation and lr 1e-4, so its larger gains do not transfer to our setup; the per-cell differences have no variance (compare our noise floors of 0.010-0.028 ROC).
- Assessment [A3] VELVET: the +0.2 F1 is a GNN + graph-node-transformer ensemble; it is not evidence about a CodeBERT text branch. Its useful message is that the graph mainly helped statement localisation, not function-level classification.
- Assessment [A4] LineVD: supports H3 ("comparable to ... without a GNN" for most graph/GNN choices). Its one positive cell is over a **frozen** CodeBERT, where the GNN is the only trainable path that can mix statements. Our backbone is fine-tuned, which removes that advantage (H1).
- Assessment [A5] DeepDFA: the strongest counter-evidence. But it differs from our design on exactly the points H2, H4 and H5 name: parser-built CFG/dataflow structure, node features built to encode dataflow facts rather than text, GNN lr 1e-3, and Big-Vul (inflated benchmark per PrimeVul). Its own sentence that text embeddings propagated along dependency edges "cannot directly capture the dataflow patterns" fits our node design (mean of CodeBERT states), which is a text encoding.
- Assessment [A6], [A7]: explicit structure helps by roughly 2 points when baked into pre-training (MRR, MAP@R), on non-VD tasks. That is a different intervention (structure seen over millions of functions) from a random-init branch trained on 456 functions.
- Assessment [A8]: the closest published analogue to our architecture (GNN on top of fine-tuned BERT output, random-init GNN, lr 2e-5). It supports H4 strongly (predicted parses at UAS 84% give "little to no gains") and argues **against** blaming H5 alone: their random-init GNN at 2e-5 did learn when the structure was gold and the data large.
- Assessment [A9], [A10]: support H1. With relative/sequential attention, explicit AST adds nothing (summarization) or helps only in some tasks (variable misuse).
- Assessment [A11]: counter-evidence for from-scratch models (structure +1.9 F1), not for a pre-trained backbone.
- Assessment [A15]: outside code, but the same comparison as ours (concatenated GNN embedding vs no-graph control) and the same two signatures: a small lift that shrinks against a matched-capacity non-graph control, and shuffled edges beating real edges. It suggests our ~9.45M extra parameters should be compared with a matched-capacity non-graph head, not only with the plain text model (inference; preprint).
- Overall: published positive cases share at least one of (parser-accurate graph, frozen or weak text encoder, purpose-built node features, larger data, higher GNN lr, structure in pre-training). Our setup has none of them, so a null result is the expected outcome rather than an anomaly.

### Gaps
- No published study adds a GNN branch to a **fine-tuned** CodeBERT for **Python** VD, or reports graph-minus-no-graph with per-fold variance. All VD hybrids read are C/C++ (Joern) except BABEL's cross-language test, whose full text is closed access (see reused note [L1]).
- GREAT's full text could not be read; its numbers beyond the abstract are not cited.
- Vul-LMGNNs, ReGVD and VELVET report single runs; whether their deltas exceed seed noise is unknown.

## Q2. Do pretrained code transformers already capture syntax and data-flow relations (so explicit edges are redundant)?

### Takeaway
Probing work on CodeBERT/GraphCodeBERT consistently finds syntax recoverable from attention and hidden states (Python included), and CodeBERT encodes data/control dependence (MCC over 60% for CDG and DDG) without any data-flow pre-training. The same studies find weaker semantic understanding (semantic equivalence, bug semantics), so "already captured" holds for the kind of relation our heuristic edges encode (shared identifiers, nesting, adjacency), not for vulnerability semantics. A transformer is formally a GNN over the complete token graph, so a line graph whose edges mostly join adjacent lines adds little connectivity that attention lacks.

### Cited Findings

**[B1] Wan et al. (ICSE 2022). VERIFIED**
- Bib: Yao Wan, Wei Zhao, Hongyu Zhang, Yulei Sui, Guandong Xu, Hai Jin. "What Do They Capture? - A Structural Analysis of Pre-Trained Language Models for Source Code." ICSE '22, Pittsburgh, 12 pages. DOI 10.1145/3510003.3510050. arXiv:2202.06840. URL read: https://arxiv.org/pdf/2202.06840 - [arXiv PDF](https://arxiv.org/pdf/2202.06840)
- Abstract findings: "(1) Attention aligns strongly with the syntax structure of code. (2) Pre-training language models of code can preserve the syntax structure of code in the intermediate representations of each Transformer layer. (3) The pre-trained models of code have the ability of inducing syntax trees of code." Models: CodeBERT, GraphCodeBERT; languages "Python, Java, and PHP". - [arXiv PDF](https://arxiv.org/pdf/2202.06840)
- Numbers: "the most aligned heads are located in the deeper layers and the concentration is as high as 67.25% (Layer 11, Head 1 in CodeBERT) and 59% (Layer 12, Head 9 in GraphCodeBERT)"; probing: "in Python, CodeBERT and GraphCodeBERT achieve the highest Spearman correlation (84% and 86%, respectively) in the 5-th layer"; "GraphCodeBERT performs better than CodeBERT, indicting that it is helpful to explicitly incorporate the syntax structure into model pre-training". Analysis truncates code at 512 tokens. - [arXiv PDF](https://arxiv.org/pdf/2202.06840)
- Caveat stated: "the current pre-trained code models do not capture well the property of the right-skewness of AST" (bias injection raises tree induction "up to 5%"). - [arXiv PDF](https://arxiv.org/pdf/2202.06840)

**[B2] Troshin & Chirkova (BlackboxNLP 2022). VERIFIED**
- Bib: Sergey Troshin, Nadezhda Chirkova. "Probing Pretrained Models of Source Codes." Proceedings of the Fifth BlackboxNLP Workshop, 2022, pp. 371-383. DOI 10.18653/v1/2022.blackboxnlp-1.31 (Crossref). arXiv:2202.08975. - [arXiv PDF](https://arxiv.org/pdf/2202.08975)
- "Our results show that pretrained models of code do contain information about code syntactic structure, the notion of namespaces, data flow, code readability and natural language-based naming. However, pretrained models show limited understanding of code semantics". Abstract: they "may fail to recognize more complex code properties such as semantic equivalence". Includes an "Edge Prediction in Data Flow Graph" probe, where "GraphCodeBERT performs best because it uses the edge prediction objective during pretraining"; "middle layers (4-10) usually provide the most informative representations". - [arXiv PDF](https://arxiv.org/pdf/2202.08975)

**[B3] Ma et al. (TOSEM 2024). VERIFIED**
- Bib: Wei Ma, Shangqing Liu, Mengjie Zhao, Xiaofei Xie, Wenhan Wang, Qiang Hu, Jie Zhang, Yang Liu. "Unveiling Code Pre-Trained Models: Investigating Syntax and Semantics Capacities." *ACM TOSEM*, 2024, pp. 1-29. DOI 10.1145/3664606 (Crossref). arXiv:2212.10017 (v3). Author list read from the arXiv v3 header. - [arXiv PDF](https://arxiv.org/pdf/2212.10017)
- "CodeBERT is also capable of encoding program semantics, despite not utilizing data flow information during pre-training. For example, the MCC reaches over 60% in CDG and DDG." Abstract: "these models are proficient in grasping code syntax ... However, their ability to encode code semantics shows more variability. CodeT5 and CodeBERT excel at capturing control and data dependencies, whereas UnixCoder performs less effectively." Probing datasets named in the passage read: Java250 and POJ-104 (not Python). - [arXiv PDF](https://arxiv.org/pdf/2212.10017)

**[B4] AST-Probe (ASE 2022). VERIFIED (abstract in PDF)**
- Bib: José Antonio Hernández López, Martin Weyssow, Jesús Sánchez Cuadrado, Houari Sahraoui. "AST-Probe: Recovering abstract syntax trees from hidden representations of pre-trained language models." ASE '22, pp. 1-11. DOI 10.1145/3551349.3556900. arXiv:2206.11719. - [arXiv PDF](https://arxiv.org/pdf/2206.11719)
- "we show that this syntactic subspace exists in five state-of-the-art pre-trained language models ... the middle layers of the models are the ones that encode most of the AST information" and its dimension "is substantially lower than those of the models' representation spaces". - [arXiv PDF](https://arxiv.org/pdf/2206.11719)

**[B5] Karmakar & Robbes (ASE 2021 NIER). VERIFIED (abstract in PDF)**
- Bib: Anjan Karmakar, Romain Robbes. "What do pre-trained code models know about code?" ASE 2021, pp. 1332-1336. DOI 10.1109/ASE51524.2021.9678927. arXiv:2108.11308. - [arXiv PDF](https://arxiv.org/pdf/2108.11308)
- Probes "for surface-level, syntactic, structural, and semantic information" on BERT, CodeBERT, CodeBERTa, GraphCodeBERT: "While GraphCodeBERT performs more consistently overall, we find that BERT performs surprisingly well on some code tasks". - [arXiv PDF](https://arxiv.org/pdf/2108.11308)

**[B6] Transformers as GNNs on the complete graph. VERIFIED (workshop paper and preprint)**
- Vijay Prakash Dwivedi, Xavier Bresson. "A Generalization of Transformer Networks to Graphs." AAAI 2021 Workshop on Deep Learning on Graphs (DLG-AAAI 2021), arXiv:2012.09699. "The original transformer was designed for Natural Language Processing (NLP), which operates on fully connected graphs representing all connections between the words in a sequence. Such architecture does not leverage the graph connectivity inductive bias, and can perform poorly when the graph topology is important and has not been encoded into the node features." - [arXiv PDF](https://arxiv.org/pdf/2012.09699)
- Chaitanya K. Joshi. "Transformers are Graph Neural Networks." arXiv:2506.22084 (Jun 2025; PREPRINT, "technical version of an article in The Gradient"). "Transformers can be viewed as message passing GNNs operating on fully connected graphs of tokens, where the self-attention mechanism capture the relative importance of all tokens w.r.t. each-other, and positional encodings provide hints about sequential ordering or structure." - [arXiv PDF](https://arxiv.org/pdf/2506.22084)

**[B7] Limits: bug semantics are not captured. VERIFIED (PREPRINT, venue not found in Crossref)**
- Benjamin Steenhoek, Md Mahbubur Rahman, Shaila Sharmin, Wei Le. "Do Language Models Learn Semantics of Code? A Case Study in Vulnerability Detection." arXiv:2311.04109. - [arXiv PDF](https://arxiv.org/pdf/2311.04109)
- "(1) better-performing models also aligned better with PVS, (2) the models failed to align strongly to PVS, and (3) the models failed to align at all to buggy paths"; input annotations "improved the models' performance in the majority of settings - 11 out of 16, with up to 9.57 points improvement in F1 score compared to conventional fine-tuning." - [arXiv PDF](https://arxiv.org/pdf/2311.04109)

### Inferences
- Assessment [B1]-[B5]: support H1 for the relations our edges encode. Our def-use edges link lines sharing a normalised identifier, our "control" edges follow indentation nesting, and next-line edges follow order; these are exactly the identifier/syntax/data-flow relations that probes recover from CodeBERT's middle layers. Since our node features are contextual states from those layers, each node already carries them before message passing.
- Assessment [B6]: supports H1 formally. With 75% of functions in one 510-token window, attention is already a GNN over the complete token graph of the function; a sparse line graph (66% of indentation edges and 46% of co-use edges join adjacent lines) restricts rather than extends that connectivity. Dwivedi & Bresson's condition for topology to matter, "has not been encoded into the node features", is not met when node features are contextual transformer states.
- Assessment [B7], [B2] semantic limits: they cut the other way. The backbone does **not** fully capture bug semantics, so there is room for extra signal; but Steenhoek et al. got it from annotating bug-relevant statements in the input, not from generic structure. Our heuristic graph carries no bug-specific information, so it cannot supply what [B7] says is missing.
- Most probing results are on Java/C/PHP corpora; Wan et al. include Python, which is the language of our target set.

### Gaps
- No probing study read measures CodeBERT's recovery of **def-use across multi-line Python statements** (the case our heuristic edges mis-handle).
- No study found that probes whether the relations a BABEL-style line graph adds are already decodable from CodeBERT line embeddings. That is a cheap local measurement (fit a linear probe for edge existence from pairs of our node features).

## Q3. Do random, shuffled or empty graphs perform similarly to real ones?

### Takeaway
Yes, often, and this is documented in graph-ML benchmarks and NLP: structure-agnostic baselines match or beat GNNs on several standard graph-classification sets (Errica et al.); a fixed GNN trained on empty graphs beats the same GNN on the real graphs by 3 to 9 accuracy points on three datasets because GNNs "overfit the given graph-structure" (Bechler-Speicher et al.); trivial trees match parse trees on 10 NLP tasks (Shi et al.). Two 2025-2026 preprints report edge randomisation leaving performance stable or improving it. I found **no** published code- or VD-specific shuffled-graph ablation, so our shuffled-graph control is itself a contribution and should be reported with the graph-free control.

### Cited Findings

**[C1] Errica et al. (ICLR 2020). VERIFIED**
- Bib: Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli. "A Fair Comparison of Graph Neural Networks for Graph Classification." ICLR 2020. arXiv:1912.09893. - [arXiv PDF](https://arxiv.org/pdf/1912.09893)
- ">47000 experiments ... five popular models across nine common benchmarks"; "by comparing GNNs with structure-agnostic baselines we provide convincing evidence that, on some datasets, structural information has not been exploited yet." "Much to our surprise, we found out that these baselines can even perform better than GNNs on some datasets". "since none of the GNNs surpasses the baseline on D&D, PROTEINS and ENZYMES, we argue that the state-of-the-art GNN models we analyzed are not able to fully exploit the structure on such datasets yet"; "small average fluctuations on these datasets are likely to be caused by other factors, such as random initializations, rather than a successful exploitation of the structure." - [arXiv PDF](https://arxiv.org/pdf/1912.09893)
- On interpretation: "if GNN performances are close to the ones of a structure-agnostic baseline, one can draw two possible conclusions: the task does not need topological information to be effectively solved, or the GNN is not exploiting graph structure adequately." - [arXiv PDF](https://arxiv.org/pdf/1912.09893)

**[C2] Bechler-Speicher et al. (ICML 2024). VERIFIED**
- Bib: Maya Bechler-Speicher, Ido Amos, Ran Gilad-Bachrach, Amir Globerson. "Graph Neural Networks Use Graphs When They Shouldn't." *Proceedings of the 41st ICML*, PMLR 235:3284-3304, 2024. arXiv:2309.04332. - [PMLR page](https://proceedings.mlr.press/v235/bechler-speicher24a.html); [arXiv PDF](https://arxiv.org/pdf/2309.04332)
- "we show that GNNs actually tend to overfit the given graph-structure. Namely, they use it even when a better solution can be obtained by ignoring it ... when the ground truth function does not use the graphs, GNNs are not guaranteed to learn a solution that ignores the graph, even with infinite data." - [arXiv abs](https://arxiv.org/abs/2309.04332)
- Table 1, same architecture trained on given graphs (GNN) vs empty graphs (GNN with no edges), accuracy: Sum 94.5 ± 0.9 vs 97.5 ± 0.7; Proteins 67.4 ± 1.9 vs 74.1 ± 2.5; Enzymes 55.2 ± 3.1 vs 64.1 ± 5.7. "The solution of GNN-empty is realizable by GNN, and the only difference between the runs is the given graph-structures. This suggests that the decreased performance of GNN is due to graph-structure overfitting." - [arXiv PDF](https://arxiv.org/pdf/2309.04332)
- Also: "regular graphs are more robust to this over-fitting". - [arXiv abs](https://arxiv.org/abs/2309.04332)

**[C3] Shi et al., trivial trees (EMNLP 2018; NLP analogue). VERIFIED**
- Bib: Haoyue Shi, Hao Zhou, Jiaze Chen, Lei Li. "On Tree-Based Neural Sentence Modeling." EMNLP 2018, pp. 4631-4641. DOI 10.18653/v1/D18-1492. arXiv:1808.09644. - [arXiv PDF](https://arxiv.org/pdf/1808.09644)
- "we replace the parsing trees with trivial trees (i.e., binary balanced tree, left-branching tree and right-branching tree) in the encoders. Though trivial trees contain no syntactic information, those encoders get competitive or even better results on all of the ten downstream tasks we investigated. This surprising result indicates that explicit syntax guidance may not be the main contributor to the superior performances of tree-based neural sentence modeling." - [arXiv PDF](https://arxiv.org/pdf/1808.09644)

**[C4] Faber et al. (2021). VERIFIED (abstract in PDF) - PREPRINT, arXiv only**
- Bib: Lukas Faber, Yifan Lu, Roger Wattenhofer. "Should Graph Neural Networks Use Features, Edges, Or Both?" arXiv:2103.06857. - [arXiv PDF](https://arxiv.org/pdf/2103.06857)
- "We find that for graph classification, a GNN is not more than the sum of its parts. We also find that, unlike features, predictions with an edge-only model do not always transfer to GNNs." - [arXiv PDF](https://arxiv.org/pdf/2103.06857)

**[C5] Karn & Jensen (2025, fake-news GNN benchmarks). VERIFIED (abstract in PDF) - PREPRINT**
- Bib: Isha Karn, David Jensen. "The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News." arXiv:2512.06638 (Dec 2025; "Preprint"). - [arXiv PDF](https://arxiv.org/pdf/2512.06638)
- "MLPs match or closely trail the performance of GNNs, with performance gaps often within 1-2% and overlapping confidence intervals ... we conduct controlled experiments where node features are shuffled or edge structures randomized. We find that performance collapses under feature shuffling but remain stable under edge randomization ... over 75% of nodes are only one hop from the root, exhibiting minimal structural diversity. In contrast, on synthetic datasets where node features are noisy and structure is informative, GNNs significantly outperform MLPs." - [arXiv PDF](https://arxiv.org/pdf/2512.06638)

**[C6] Maganti (2026). See [A15]. PREPRINT**
- "randomly shuffled edges outperform the real transaction graph by 8.9 F1 points" (10 seeds). - [arXiv PDF](https://arxiv.org/pdf/2604.19514)

**[C7] Duggireddy (2026), what randomisation measures. PARTIAL (abstract only) - alphaXiv-only, not found on arXiv; not peer-reviewed**
- Prahas Duggireddy. "Dependence Is Not Advantage: What Randomizing the Graph Does Not Measure." alphaXiv, submitted 7 Aug 2026 (an arXiv title search returned nothing). "randomization measures how a fixed procedure depends on the observed edge arrangement, whereas improvement requires comparison with a specified graph-free baseline"; recommends "reporting the intact score, graph-free baseline, survival ratio, and sampler diagnostics together." - [alphaXiv page](https://www.alphaxiv.org/abs/2608.dependence-is-not-advantage)

### Inferences
- Assessment [C1], [C2]: support H3 directly. Our pattern (real graph within noise of no-graph, shuffled at least as good) is the documented signature of "structure not exploited / graph overfitting". [C2]'s mechanism also explains why a real graph can be *worse* than a shuffled one: shuffled co-use/indentation edges in short functions approach a near-regular random graph, which [C2] finds more robust to overfitting (inference; [C2] tests regular graphs, not shuffled code graphs).
- Assessment [C3]: the closest "fake structure works as well" result for a sequence domain; it supports reading our shuffled-graph result as "the branch's value, if any, is extra pooling/capacity, not syntax".
- Assessment [C5]: its mechanism matches our graph statistics. Their graphs are shallow (75% of nodes one hop from the root); ours are dominated by adjacent-line edges (66% / 46%), which a sequence model already sees. Both make structure low-information.
- Assessment [C7] (weak source): methodologically consistent with how the project already reports (graph vs no-graph vs shuffled, paired). It argues that "real < shuffled" does not by itself prove harm; the no-graph comparator is the one that measures advantage.
- Errica et al.'s two readings ("task does not need topology" vs "GNN does not exploit it") map onto H2 (pairs have identical graphs, so the task gives topology nothing to do) vs H5/H6 (the branch cannot exploit it). The 63%-identical-graph measurement favours the first reading for at least 63% of same-length pairs.

### Gaps
- **No code or VD paper found that reports a random/shuffled-edge ablation** for a GNN over program graphs (searched 2026-10-05 with several phrasings). The VD literature reports edge-type ablations (e.g. LineVD CDG vs PDG) but not random-edge controls.
- Homophily work (Ma et al., "Is Homophily a Necessity for Graph Neural Networks?", ICLR 2022, arXiv:2106.06134, abstract read) concerns node classification; it does not transfer cleanly to graph-level labels on vulnerable/patched pairs and is not used as evidence here.

## Q4. Do vulnerable and patched versions differ so little that structure cannot discriminate them?

### Takeaway
Security fixes are small and local: median 7 changed LOC versus 16 for other bug fixes, only 6% over 100 lines, and 59% confined to one function (Li & Paxson, CCS 2017, >4,000 fixes); npm fixes have a median of 10 LoC. SVEN's own authors note that "only the edited code in these fixes is decisive for security", and their example fix is wrapping a value in `markupsafe.escape`, an in-line call change that a line graph with normalised identifiers does not see. Pair-level evaluations show transformer detectors themselves barely separate a function from its patch (VulnPatchPairs accuracy 0.294-0.527; PrimeVul pair-wise correct 1.06-3.01%), and truncation makes 27% of PrimeVul pairs identical. A graph that is identical for 63% of same-length pairs can add discrimination only through its node features, which are the text encoder's states.

### Cited Findings

**[D1] Li & Paxson (CCS 2017). VERIFIED**
- Bib: Frank Li, Vern Paxson. "A Large-Scale Empirical Study of Security Patches." CCS '17, Dallas, pp. 2201-2215. DOI 10.1145/3133956.3134072 (Crossref). URL read: https://faculty.cc.gatech.edu/~frankli/papers/li-ccs2017.pdf - [author PDF](https://faculty.cc.gatech.edu/~frankli/papers/li-ccs2017.pdf)
- Scope: "more than 4,000 bug fixes for over 3,000 vulnerabilities that affected a diverse set of 682 open-source software projects". - [author PDF](https://faculty.cc.gatech.edu/~frankli/papers/li-ccs2017.pdf)
- "security commits overall are statistically significantly less complex and smaller than non-security bug patches (p ≈ 0). The median security commit diff involved 7 LOC compared to 16 LOC for non-security bug fixes. Approximately 20% of non-security patches had diffs with over 100 lines changed, while this occurred in only 6% of security commits." "59% of security changes resided in a single function, compared to 42% of other bug fixes." "Nearly 78% of security commits did not delete any code". Mostly C/C++/PHP projects; not Python-specific. - [author PDF](https://faculty.cc.gatech.edu/~frankli/papers/li-ccs2017.pdf)

**[D2] Chinthanet et al., npm fixes (EMSE 2021). VERIFIED**
- Bib: Bodin Chinthanet, Raula Gaikovina Kula, Shane McIntosh, Takashi Ishio, Akinori Ihara, Kenichi Matsumoto. "Lags in the release, adoption, and propagation of npm vulnerability fixes." *Empirical Software Engineering*, 2021. DOI 10.1007/s10664-021-09951-x. arXiv:1907.03407. - [arXiv PDF](https://arxiv.org/pdf/1907.03407)
- "the fix itself tends to contain only a few lines of code, i.e., median of 10 LoC" (231 vulnerabilities), against a fixing release median of 219 LoC. - [arXiv PDF](https://arxiv.org/pdf/1907.03407)

**[D3] SVEN (CCS 2023): the target dataset's own description of its pairs. VERIFIED**
- Bib: Jingxuan He, Martin Vechev. "Large Language Models for Code: Security Hardening and Adversarial Testing." CCS '23, pp. 1865-1879. DOI 10.1145/3576915.3623175. arXiv:2302.05319. - [arXiv PDF](https://arxiv.org/pdf/2302.05319)
- "We make the key observation that only the edited code in these fixes is decisive for security, while the unchanged code is neutral." Example: "adding a call to the function markupsafe.escape turns the program from unsafe to secure". They compute diffs at three levels and use "character-level masks for secure programs and line-level masks for unsafe programs". - [arXiv PDF](https://arxiv.org/pdf/2302.05319)

**[D4] Risse & Böhme (USENIX Security 2024). VERIFIED (reused)**
- Bib: Niklas Risse, Marcel Böhme. "Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection." USENIX Security 2024. arXiv:2306.17193. - [arXiv PDF](https://arxiv.org/pdf/2306.17193)
- "state-of-the-art ML4VD techniques are unable to distinguish vulnerable functions from their patches"; on VulnPatchPairs "the accuracy drops dramatically (between 0.294 and 0.527) ... On average, the accuracy is worse than random guessing." Six transformer models, no GNN. - [arXiv PDF](https://arxiv.org/pdf/2306.17193)

**[D5] PrimeVul pair-wise evaluation (ICSE 2025). VERIFIED (reused)**
- Bib: Yangruibo Ding et al. "Vulnerability Detection with Code Language Models: How Far Are We?" ICSE 2025, pp. 1729-1741. DOI 10.1109/ICSE55347.2025.00038. arXiv:2403.18624. - [arXiv PDF](https://arxiv.org/pdf/2403.18624)
- Pair-wise correct (P-C) on PrimeVul: CodeT5 1.06, CodeBERT 1.77, UniXcoder 1.60, StarCoder2 2.30, CodeGen2.5 3.01 (%). "these models make decisions primarily based on textual similarity, without considering the underlying root causes or fixes of the vulnerabilities." - [arXiv PDF](https://arxiv.org/pdf/2403.18624)

**[D6] RevisitVD (AsiaCCS 2026). VERIFIED (reused)**
- Bib: Youpeng Li, Weiliang Qi, Xuyu Wang, Fuxun Yu, Xinda Wang. "Revisiting Pre-trained Language Models for Vulnerability Detection." arXiv:2507.16887 (accepted at AsiaCCS 2026 per arXiv comment). - [arXiv PDF](https://arxiv.org/pdf/2507.16887)
- "By analyzing 5,480 patch pairs in PrimeVul, we find that 1,473 pairs (27%) have this issue" (pair identical after truncation, opposite labels). - [arXiv PDF](https://arxiv.org/pdf/2507.16887)

**[D7] BABEL's heuristic edges depend on line formatting. VERIFIED for code/README (reused [L1])**
- The README recommends clang-format with a large `ColumnLimit` so each line holds as many statements as possible, "suitable for our subsequent extraction of inter-line dependencies based on heuristic rules"; the identifier symbolizer carries a C/C++ keyword list. - [BABEL README](https://github.com/gdufsnlp/BABEL)

### Inferences
- Assessment [D1]-[D3]: support H2 strongly. A median 7-line fix that usually updates lines in place (78% delete nothing) leaves the set of lines and their indentation unchanged; when the edit is inside a call argument or a string literal, as in SVEN's `markupsafe.escape` example, the normalised-identifier co-use and def-use edges are unchanged as well. That is the mechanism behind the measured 63% of same-length pairs with byte-identical graphs. For those pairs the graph branch can differ only through node features, which are the same CodeBERT states the text branch already pools, so it is a second text pooling head at best.
- Assessment [D4]-[D6]: show the hard part of the task (separating a function from its patch) is unsolved for transformers too. They do not show that graphs would do better; no GNN was evaluated there. Combined with H2, they predict that neither branch gets pair-discriminative signal from structure.
- Assessment [D7]: supports H4. The original BABEL relies on a formatter to put whole statements on one line; Python has no equivalent normalisation in our pipeline, so multi-line calls (e.g. data flowing into `.format(...)`) split one statement over several nodes and the def-use rule misses the sink.
- Note on SVEN as target: under random 60/20/20 splits, both versions of a pair can land in train and test (by design in this project), so structure that is identical within a pair cannot even be used to memorise the pair label.

### Gaps
- No published statistic for **Python** fix size (lines/tokens changed) or for SVEN specifically was found; Li & Paxson's projects are mostly C/C++/PHP. The 63% identical-graph figure and the "changes inside literals/arguments" observation are local measurements without a published counterpart.
- No paper measures how often a vulnerable/patched pair yields identical program graphs under any graph construction (parser-based or heuristic).

## Q5. Over-smoothing, mean aggregation of contextual features, and learning rate for random-init modules

### Takeaway
Over-smoothing is a deep-stack effect (Li et al. 2018; Oono & Suzuki 2020) and AMPLE finds 2 GCN layers best in VD, so with 2 R-GCN layers it is an unlikely main cause. Mean aggregation is a more plausible loss of signal: GIN shows mean aggregators cannot separate multisets with the same distribution, and our node features are already means of contextual token states, so a 1-3 token edit is diluted before message passing. On optimisation, every VD graph branch read uses a GNN learning rate of 1e-4 to 1e-3 (or freezes the PLM), 5-50x our 2e-5; fine-tuning theory warns that random-init heads distort pretrained features and few-sample BERT fine-tuning is under-trained at default iteration counts. But Sachan et al. trained a random-init GNN on BERT at 2e-5 and it learned with gold structure on large data, so lr alone does not explain a null when the structure is uninformative.

### Cited Findings

**[E1] Li, Han, Wu (AAAI 2018). VERIFIED**
- Bib: Qimai Li, Zhichao Han, Xiao-Ming Wu. "Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning." AAAI 2018. DOI 10.1609/aaai.v32i1.11604. arXiv:1801.07606. - [arXiv PDF](https://arxiv.org/pdf/1801.07606)
- "the graph convolution of the GCN model is actually a special form of Laplacian smoothing, which is the key reason why GCNs work, but it also brings potential concerns of oversmoothing with many convolutional layers." - [arXiv PDF](https://arxiv.org/pdf/1801.07606)

**[E2] Oono & Suzuki (ICLR 2020). VERIFIED**
- Bib: Kenta Oono, Taiji Suzuki. "Graph Neural Networks Exponentially Lose Expressive Power for Node Classification." ICLR 2020. arXiv:1905.10947. - [arXiv PDF](https://arxiv.org/pdf/1905.10947)
- "it is known that they do not improve (or sometimes worsen) their predictive performance as we pile up many layers"; under spectral conditions a GCN's "output exponentially approaches the set of signals that carry information of the connected components and node degrees only for distinguishing nodes." - [arXiv PDF](https://arxiv.org/pdf/1905.10947)

**[E3] AMPLE layer ablation (ICSE 2023). VERIFIED (reused)**
- "EA-GCN with 2 layers obtains the highest accuracy of 62.16% and F1 score of 66.94%. As the number of layers continues to increase, the accuracy and F1 score decrease. We suppose that the GCN encounters an over-smoothing issue". DOI 10.1109/ICSE48619.2023.00191. - [arXiv PDF](https://arxiv.org/pdf/2302.04675)

**[E4] Xu et al., GIN (ICLR 2019). VERIFIED**
- Bib: Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. "How Powerful are Graph Neural Networks?" ICLR 2019. arXiv:1810.00826. - [arXiv PDF](https://arxiv.org/pdf/1810.00826)
- "certain popular injective set functions, such as the mean aggregator, are not injective multiset functions"; "Any mean aggregator maps X1 and X2 to the same embedding, because it simply takes averages over individual element features"; "The mean aggregator may perform well if, for the task, the statistical and distributional information in the graph is more important than the exact structure." - [arXiv PDF](https://arxiv.org/pdf/1810.00826)

**[E5] Kumar et al., random heads distort pretrained features (ICLR 2022). VERIFIED**
- Bib: Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang. "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution." ICLR 2022 (Oral, per arXiv comment). arXiv:2202.10054. - [arXiv PDF](https://arxiv.org/pdf/2202.10054)
- "the OOD error of fine-tuning is high when we initialize with a fixed or random head - this is because while fine-tuning learns the head, the lower layers of the neural network change simultaneously and distort the pretrained features"; LP-FT gives "1% better ID, 10% better OOD than full fine-tuning". Vision benchmarks, linear heads. - [arXiv PDF](https://arxiv.org/pdf/2202.10054)

**[E6] ULMFiT discriminative fine-tuning (ACL 2018). VERIFIED**
- Bib: Jeremy Howard, Sebastian Ruder. "Universal Language Model Fine-tuning for Text Classification." ACL 2018, pp. 328-339. DOI 10.18653/v1/P18-1031. arXiv:1801.06146. - [arXiv PDF](https://arxiv.org/pdf/1801.06146)
- "As different layers capture different types of information (Yosinski et al., 2014), they should be fine-tuned to different extents ... Instead of using the same learning rate for all layers of the model, discriminative fine-tuning allows us to tune each layer with different learning rates." - [arXiv PDF](https://arxiv.org/pdf/1801.06146)

**[E7] Zhang et al., few-sample BERT fine-tuning (ICLR 2021). VERIFIED**
- Bib: Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger, Yoav Artzi. "Revisiting Few-sample BERT Fine-tuning." ICLR 2021. arXiv:2006.05987. Author list read from the PDF header. - [arXiv PDF](https://arxiv.org/pdf/2006.05987)
- Causes of instability include "the prevalent practice of using a pre-determined, and small number of training iterations"; "Training longer can improve over the three-epochs setup most of the time, in terms of both performance and stability. This is more pronounced on the 1k downsampled datasets." - [arXiv PDF](https://arxiv.org/pdf/2006.05987)

**[E8] Learning rates actually used for VD graph branches. VERIFIED (sources above)**
- ReGVD lr grid 1e-4 / 5e-4 / 1e-3 ([A1]); Vul-LMGNNs 1e-4, 20 epochs ([A2]); VELVET 1e-4 pre-train, 1e-5 fine-tune ([A3]); DeepDFA 1e-3 ([A5]); LineVD freezes CodeBERT ([A4]); Sachan et al. 2e-5 for everything ([A8]). - [ReGVD](https://arxiv.org/pdf/2110.07317); [Vul-LMGNNs](https://arxiv.org/pdf/2404.14719); [VELVET](https://arxiv.org/pdf/2112.10893); [DeepDFA](https://arxiv.org/pdf/2212.08108); [LineVD](https://arxiv.org/pdf/2203.05181); [Sachan](https://arxiv.org/pdf/2008.09084)

### Inferences
- Assessment [E1]-[E3]: argue **against** over-smoothing as a main cause. With 2 R-GCN layers we are at AMPLE's best depth, and the theory concerns many stacked layers. Smoothing still matters in a milder form: [E1] says one convolution already "mixes the features of a vertex and its nearby neighbors", and with 66% of indentation edges joining adjacent lines, two layers average each line with its immediate neighbours, which dilutes a one-line change further (inference).
- Assessment [E4]: supports H6. Mean-pooling a line's contextual token states already compresses a 1-3 token edit into a small shift of a 768-dim mean, and the downstream LSTM + max-pool then has to recover a difference the text branch sees at token resolution. GIN's remark that mean works when "distributional information ... is more important than the exact structure" describes a task where the graph is unnecessary.
- Assessment [E5]-[E8]: support H5 as a contributing factor. Published graph branches use 5-50x our learning rate or freeze the backbone; ~9.45M random-init parameters at 2e-5 for a few epochs on ~456 functions is under-trained by every precedent read (inference; none of these papers ran our exact setting). [E5] adds a second risk: a large random branch trained jointly can perturb the pretrained text features, which would show up as sign flips across configurations rather than a consistent loss.
- Counter-assessment [A8]: Sachan et al.'s random-init GNN on BERT at 2e-5 did learn when parses were gold and data large. So H5 explains why a weak signal might not be picked up, but not a null where the structure is identical within 63% of pairs (H2). The most defensible ordering of explanations is H2 (structure non-discriminative) and H1 (redundant with attention) first, H3/H4 as consistent corroboration (shuffled = real; heuristic edges noisy), H5/H6 as reasons the branch could not compensate, and over-smoothing last.

### Gaps
- No paper found that measures the effect of a separate (higher) learning rate for a GNN branch on top of a **fine-tuned** code PLM; the VD papers fix one lr without ablating it. Testing lr 1e-4 to 1e-3 for the graph branch with the backbone at 2e-5 would be the direct check (the project's memory already records "new modules need their own learning rate").
- No study found on mean-pooling contextual subword states into statement/line nodes versus first-token or attention pooling for VD.
