# Why SAM/ASAM helps fine-tuning and transfer: evidence for the MultiVD two-phase results

Scope: published evidence (up to Oct 2026) for and against seven candidate explanations of the measured ASAM pattern in MultiVD. The setup is a CodeBERT multi-window classifier. Phase 1 trains on source-language vulnerability data. Phase 2 fine-tunes on 456 SVEN Python functions with ASAM rho=0.5, eta=0.01, over 5 folds with deltas paired by fold.

Short names for the measured pattern used below:
- **P1**: JS Phase-1 checkpoint + ASAM in Phase 2 gives ROC-AUC +0.018 to +0.030, mostly 5/5 folds, in two blocks.
- **P2**: no Phase 1 (straight from CodeBERT) gives an inconsistent ASAM effect: +0.030 (5/5) in one block, +0.001 (2/2) in another (different code, no LR warmup).
- **P3**: C/C++(+JS) Phase-1 checkpoints give a mixed ASAM effect (-0.014 to +0.027).
- **P4**: interaction. Phase-1 transfer helps more under ASAM (e.g. +0.016 ROC with AdamW vs +0.043 with ASAM, same source).
- **P5**: ASAM moves ranking metrics more consistently than F1@0.5 (190 cells: ROC +0.0037, 119/190, p=0.0006; F1@0.5 +0.0015, 100/190, p=0.51).
- **P6**: target labels (SVEN, ~94% accurate) are cleaner than source labels (CleanVul, PrimeVul).

Status legend: **VERIFIED** = I read the quoted text and numbers in the primary full text (PDF downloaded from the URL given and converted with pdftotext). **PARTIAL** = I read only the abstract. **UNVERIFIED** = not read in a primary source. Venue and page numbers come from the publisher index pages listed.

### Citation registry

| Key | Authors | Exact title | Venue, year, pages | DOI / arXiv | URL actually read | Status |
|---|---|---|---|---|---|---|
| Foret21 | Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur | Sharpness-Aware Minimization for Efficiently Improving Generalization | ICLR 2021 (PDF header "Published as a conference paper at ICLR 2021") | arXiv:2010.01412 (v3) | https://arxiv.org/pdf/2010.01412 | VERIFIED |
| Kwon21 | Jungmin Kwon, Jeongseop Kim, Hyunseo Park, In Kwon Choi | ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks | ICML 2021, PMLR 139:5905-5914 | arXiv:2102.11600 (v3) | https://arxiv.org/pdf/2102.11600 ; pages from https://proceedings.mlr.press/v139/ (paper page https://proceedings.mlr.press/v139/kwon21b.html) | VERIFIED |
| Bahri22 | Dara Bahri, Hossein Mobahi, Yi Tay | Sharpness-Aware Minimization Improves Language Model Generalization | ACL 2022 (Vol. 1: Long Papers), Dublin, pp. 7360-7371 | DOI 10.18653/v1/2022.acl-long.508 ; arXiv:2110.08529 (v2) | https://arxiv.org/pdf/2110.08529 ; https://aclanthology.org/2022.acl-long.508/ | VERIFIED |
| Baek24 | Christina Baek, Zico Kolter, Aditi Raghunathan | Why is SAM Robust to Label Noise? | ICLR 2024 (PDF header) | arXiv:2405.03676 (v1) | https://arxiv.org/pdf/2405.03676 | VERIFIED |
| Springer24 | Jacob Mitchell Springer, Vaishnavh Nagarajan, Aditi Raghunathan | Sharpness-Aware Minimization Enhances Feature Quality via Balanced Learning | ICLR 2024 (PDF header) | arXiv:2405.20439 (v1) | https://arxiv.org/pdf/2405.20439 | VERIFIED |
| Andr23LR | Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, Nicolas Flammarion | Sharpness-Aware Minimization Leads to Low-Rank Features | NeurIPS 2023 (Advances 36), pp. 47032-47051 | arXiv:2305.16292 (v2) | https://arxiv.org/pdf/2305.16292 ; https://papers.nips.cc/paper_files/paper/2023/hash/92dd1adab39f362046f99dfe3c39d90f-Abstract-Conference.html | VERIFIED |
| Watts26 | Ishaan Watts, Catherine Li, Sachin Goyal, Jacob Mitchell Springer, Aditi Raghunathan | Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting | ICML 2026 per arXiv comment "accepted to ICML2026". Proceedings pages not seen, so this is the **preprint** version | arXiv:2605.02105 (v1, 4 May 2026) | https://arxiv.org/pdf/2605.02105 ; arXiv API metadata | VERIFIED (content); venue from arXiv comment only |
| Mehta23 | Sanket Vaibhav Mehta, Darshan Patil, Sarath Chandar, Emma Strubell | An Empirical Investigation of the Role of Pre-training in Lifelong Learning | JMLR 24(214):1-50, 2023 | arXiv:2112.09153 (v2) | https://arxiv.org/pdf/2112.09153 ; https://jmlr.org/papers/v24/ (index entry, paper id 22-0496) | VERIFIED |
| Cha21 | Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, Sungrae Park | SWAD: Domain Generalization by Seeking Flat Minima | NeurIPS 2021 (Advances 34), pp. 22405-22418 | arXiv:2102.08604 (v4) | https://arxiv.org/pdf/2102.08604 ; https://papers.nips.cc/paper_files/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html | VERIFIED |
| Liu23 | Hong Liu, Sang Michael Xie, Zhiyuan Li, Tengyu Ma | Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models | ICML 2023, PMLR 202:22188-22214 | arXiv:2210.14199 (v1) | https://arxiv.org/pdf/2210.14199 ; https://proceedings.mlr.press/v202/ (paper page liu23ao) | VERIFIED |
| Tan26 | Chengli Tan, Yubo Zhou, Haishan Ye, Guang Dai, Junmin Liu, Zengjie Song, Jiangshe Zhang, Zixiang Zhao, Yunda Hao, Yong Xu | Towards Understanding The Calibration Benefits of Sharpness-Aware Minimization | ICLR 2026 (PDF header; arXiv comment "ICLR2026") | arXiv:2505.23866 (v2) | https://arxiv.org/pdf/2505.23866 | VERIFIED |
| Andr23ML | Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, Nicolas Flammarion | A Modern Look at the Relationship between Sharpness and Generalization | ICML 2023, PMLR 202:840-902 | arXiv:2302.07011 (v2) | https://arxiv.org/pdf/2302.07011 ; https://proceedings.mlr.press/v202/andriushchenko23a.html (index) | VERIFIED |
| Kaddour22 | Jean Kaddour, Linqing Liu, Ricardo Silva, Matt J. Kusner | When Do Flat Minima Optimizers Work? | NeurIPS 2022 (Advances 35), pp. 16577-16595 | arXiv:2202.00661 (v5) | https://arxiv.org/pdf/2202.00661 ; https://papers.nips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html | VERIFIED |
| AF22 | Maksym Andriushchenko, Nicolas Flammarion | Towards Understanding Sharpness-Aware Minimization | ICML 2022, PMLR 162:639-668 | arXiv:2206.06232 (v1) | https://arxiv.org/pdf/2206.06232 ; https://proceedings.mlr.press/v162/andriushchenko22a.html (index) | VERIFIED |
| TRAM24 | Tom Sherborne, Naomi Saphra, Pradeep Dasigi, Hao Peng | TRAM: Bridging Trust Regions and Sharpness Aware Minimization | ICLR 2024 (spotlight) | arXiv:2310.03646 (v2) | https://arxiv.org/pdf/2310.03646 | VERIFIED |
| FSAM22 | Qihuang Zhong, Liang Ding, Li Shen, Peng Mi, Juhua Liu, Bo Du, Dacheng Tao | Improving Sharpness-Aware Minimization with Fisher Mask for Better Generalization on Language Models | Findings of EMNLP 2022, Abu Dhabi, pp. 4064-4085 | DOI 10.18653/v1/2022.findings-emnlp.300 | https://aclanthology.org/2022.findings-emnlp.300.pdf ; https://aclanthology.org/2022.findings-emnlp.300/ | VERIFIED |
| Ju22 | Haotian Ju, Dongyue Li, Hongyang R. Zhang | Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees | ICML 2022, PMLR 162:10431-10461 | arXiv:2206.02659 (v6) | https://arxiv.org/pdf/2206.02659 | VERIFIED (abstract, intro, bound statement) |
| Chen22 | Xiangning Chen, Cho-Jui Hsieh, Boqing Gong | When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations | ICLR 2022 (spotlight, per arXiv comment) | arXiv:2106.01548 (v3) | https://arxiv.org/pdf/2106.01548 | VERIFIED (transfer appendix and Table 10) |
| Sadr23 | Ildus Sadrtdinov, Dmitrii Pozdeev, Dmitry Vetrov, Ekaterina Lobacheva | To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning | NeurIPS 2023 (per arXiv comment; pages not checked) | arXiv:2303.03374 (v3) | https://arxiv.org/pdf/2303.03374 | PARTIAL (abstract only) |
| STAB26 | Shuyu Chang, Haiping Huang, Yanjun Zhang, Yujin Huang, Fu Xiao, Leo Yu Zhang | Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation | **preprint**, arXiv Feb 2026 | arXiv:2602.11213 | arXiv API abstract (https://export.arxiv.org/api/query?id_list=2602.11213) | PARTIAL (abstract only) |

Local context reused (read-only):
- The project note `/drive1/cuongtm/ntat/MultiVD/RESEARCH_2026-09-06_recadam.md` (line 91) records ASAM rho in {0.1, 0.2, 0.5} in Phase 2 on CodeT5+ (t5p) as "not above the noise floor at n=15" (FACTS §18).
- `/drive1/cuongtm/ntat/MultiVD/RESEARCH_2026-08-20_0959.md` (line 200) summarizes Watts26. I re-verified its quote and venue against the arXiv PDF and metadata above.

---

## Q1. Flat minima and transferability/robustness: do flatter solutions transfer better or generalize better under distribution shift?

### Takeaway
There is real but mixed evidence. Theory (SWAD) and several experiments link flatness to OOD/domain generalization and to downstream transfer. In most of these experiments, though, the flatness belongs to the **upstream/source** model, not to a fine-tuning step on the target. A large study (Andr23ML) finds that sharpness does **not** predict generalization for BERT fine-tuning, and that flatter CLIP fine-tunes do worse OOD. "Flatter = transfers better" is therefore a hypothesis, not an established law, for a CodeBERT fine-tune.

### Cited Findings
- **SWAD theory (VERIFIED).** Cha21 abstract: "we theoretically show that finding flat minima results in a smaller domain generalization gap." SWAD gives "+1.6% averagely on out-of-domain accuracy" on PACS, VLCS, OfficeHome, TerraIncognita and DomainNet. - [Cha21, arXiv PDF](https://arxiv.org/pdf/2102.08604)
- **SAM inside SWAD's DomainBed comparison (VERIFIED).** Out-of-domain averages (Table 4) are ERM 63.3 vs SAM 64.5 (+1.2 pp). SAM is **worse** on TerraIncognita: 46.1±1.8 (ERM) vs 43.3±0.7 (SAM). In the in-domain/out-of-domain split study, "SAM, another method for seeking flat minima, slightly increases both in-domain and out-of-domain performances but the out-of-domain performance is not statistically significant." SAM was run "with ρ = 0.05". - [Cha21](https://arxiv.org/pdf/2102.08604)
- **SAM-trained representations transfer better under linear probing (VERIFIED).** Springer24, Appendix F (DomainBed): a classifier is trained on all source domains with SAM or SGD, then a linear probe is fit on the target domain. SAM beats SGD on all 12 target domains, for example:
  - OfficeHome Clipart 0.716 to 0.745
  - PACS Cartoon 0.897 to 0.921
  - VLCS SUN09 0.748 to 0.776

  Quote: "We find that SAM outperforms SGD on all three datasets, confirming that SAM improves the representation quality of the neural network in a variety of domain transfer settings." Rho was swept over {0, 0.03, 0.05, 0.1}. - [Springer24](https://arxiv.org/pdf/2405.20439)
- **Flatness of the pretrained LM predicts downstream (VERIFIED).** Liu23 abstract: "flatness of the model is well-correlated with downstream performance where pre-training loss is not" and "among the models with the minimal pre-training loss, the flattest model transfers to downstream tasks." Flatness is measured "by the trace of Hessian of the loss". This holds for both fine-tuning and linear probing evaluations, on simplified datasets. - [Liu23](https://arxiv.org/pdf/2210.14199)
- **SAM pretraining improves downstream transfer even when fine-tuning without SAM (VERIFIED).** Chen22, Appendix: "Note that we do not employ SAM during fine-tuning." Table 10 average transfer accuracy (CIFAR-10/100, Flowers, Pets):
  - ViT-S/16: 90.0 to 92.6
  - ViT-B/16: 91.5 to 93.2
  - Mixer-B/16: 86.1 to 91.7 - [Chen22](https://arxiv.org/pdf/2106.01548)
- **ASAM as a fine-tuning optimizer for cross-lingual transfer (VERIFIED).** TRAM24, Table 5: XLM-RoBERTa Base trained on English MultiNLI, then zero-shot on 14 XNLI languages, mean of 20 seeds.
  - Adam: EN 83.9, zero-shot avg 72.9
  - SAM: 84.8 / 73.7
  - ASAM (rho=0.5): 85.0 / 74.0

  ASAM gives **no** gain for ImageNet to CIFAR-100 fine-tuning (Table 3: SGD 87.97±0.12, ASAM 87.97±0.08). The abstract notes "generalization during fine-tuning is often more dependent on the transferability of representations in the function space." - [TRAM24](https://arxiv.org/pdf/2310.03646)
- **Contradicting evidence (VERIFIED).** Andr23ML abstract: "we observe that sharpness does not correlate well with generalization but rather with some training parameters like the learning rate"; "in multiple cases, we observe a consistent negative correlation of sharpness with out-of-distribution error implying that sharper minima can generalize better."
  - On 50 BERT models fine-tuned on MNLI (varying only the seed), correlation with MNLI/HANS test error "is weak and does not exceed 0.04".
  - "sharpness is not useful to distinguish different solutions found by fine-tuning BERT on MNLI. All this evidence suggests that the intuitive ideas about the generalization benefits of flat minima are not supported in the modern settings."
  - "CLIP models fine-tuned on ImageNet suggest that flatter solutions consistently generalize worse on OOD data." - [Andr23ML](https://arxiv.org/pdf/2302.07011)
- **Heterogeneity across datasets (VERIFIED).** Kaddour22 findings: "1. Datasets matter" and "2. Architectures matter". For one architecture SAM gives ">1.30%" while a sibling architecture on the same dataset gets "−0.31%". - [Kaddour22](https://arxiv.org/pdf/2202.00661)

### Inferences
- **Fit to the measured pattern: weak to moderate.** Most positive evidence (Liu23, Chen22, Springer24, Watts26 in Q5) concerns flatness of the **source/upstream** solution, which is then reused downstream. In MultiVD, ASAM is applied only in **Phase 2**, on a target test set drawn from the same distribution as the target train set (random 60/20/20 split of SVEN). So the Phase-2 ASAM gain is in-distribution small-sample generalization, not OOD transfer.
- The closest analogue is TRAM24's XNLI result: ASAM as the fine-tuning optimizer of a RoBERTa-family encoder, with rho=0.5 like MultiVD, gives +1.1 points zero-shot. That shows ASAM in the *downstream* phase can help a pretrained encoder. But that target set is large (MultiNLI), and the same paper shows a null for ImageNet to CIFAR-100.
- Andr23ML directly contradicts using "flatness" as the *reason* for gains in BERT fine-tuning. Any write-up should present flatness as the method's motivation, not as a measured mechanism, unless MultiVD measures sharpness itself (the repo has `src/measure_sharpness.py`).
- P3 (mixed across sources) is consistent with Kaddour22 "datasets matter" and with SAM's loss on TerraIncognita in Cha21. Flat-minima optimizers do not help uniformly across data distributions.

### Gaps
- I found no paper measuring whether a SAM/ASAM **fine-tune on the target** (rather than SAM on the source) makes the target model generalize better under a *source-to-target language* shift for code.
- I found no study of sharpness vs generalization specifically for CodeBERT or other code encoders.

---

## Q2. SAM for fine-tuning pretrained language models on small data (Bahri et al.), and any code-model / SE / vulnerability-detection use

### Takeaway
Bahri22 shows clear SAM gains when fine-tuning T5 from public checkpoints, and the gain is often larger with little data. But each run was done **once** (single seed, best checkpoint per task-metric pair). Multi-seed studies on BERT-family encoders (FSAM22, Kaddour22) find small and inconsistent average gains from vanilla SAM. I found **no** paper that uses SAM/ASAM to train a code model for vulnerability detection or for any SE classification task. The only SAM + code-model paper found uses SAM to build a backdoor attack.

### Cited Findings
- **Bahri22 abstract (VERIFIED).** SAM "can substantially improve the generalization of language models without much computational overhead ... with particularly large gains when training data for these tasks is limited." Contribution 2: "The improvement brought by SAM often increases with less labeled training data ... We test this by subsampling the training splits of CBQA and SuperGLUE datasets at rates ranging from 2% to 80%." - [Bahri22](https://arxiv.org/pdf/2110.08529)
- **Bahri22 full-data numbers (VERIFIED).** These are T5.1.1 SuperGLUE dev overall scores:
  - Small: 67.7 to 68.4 (rho 0.05)
  - Base: 75.3 to 78.5 (rho 0.15)
  - Large: 84.3 to 84.6
  - XL: 87.2 to 89.1

  Text: "For Base and XL sizes on SuperGLUE, SAM brings 4.2% and 2.1% relative gains in overall score respectively, while the gain for Large on GLUE is 2.4%." - [Bahri22](https://arxiv.org/pdf/2110.08529)
- **Bahri22 low-data numbers (VERIFIED).** Table 6, SuperGLUE with 5% of training data:
  - Small: 50.2 to 51.9
  - Base: 52.9 to 56.7
  - Large: 62.8 to 64.3
  - XL: 75.9 to 77.0

  Text: "a whopping 7.2% relative improvement on the Base model on 5% SuperGLUE and a relative 8.86%/16.6% to F1/EM on Natural Questions". Figure 1: "SAM's improvement is consistent across data size regimes and that the relative improvement is often largest in the ballpark of 20%." Per-task cells are not all positive. For example, Base WiC at 5% goes 57.2 to 55.3, and Small CoPA at full data goes 67.0 to 61.0. - [Bahri22](https://arxiv.org/pdf/2110.08529)
- **Bahri22 protocol caveats (VERIFIED).** "We run each experiment once, due to resource constraints, and we take the best checkpoint ... across training steps", and "we report the best checkpoint for each task-metric pair ... individually."
  - Rho search: "[0.02, 0.05, 0.1, 0.15, 0.2, 0.3] a single time only when fine-tuning on SuperGLUE ... 0.05 ... for T5.1.1 small models, and 0.15 for the Base, Large, and XL variants". For mT5 on TyDiQA "a smaller ρ was necessary".
  - Optimizer: AdaFactor, lr 1e-3, batch 128.
  - Overhead: SAM is "about 25% slower" with ascent micro-batch = 1/4 of the batch. - [Bahri22](https://arxiv.org/pdf/2110.08529)
- **SAM on encoder PLMs is small and inconsistent (VERIFIED).** FSAM22 Table 1 (Adam vs Adam+SAM, average over CoLA/MRPC/STS-B/RTE/CB/BoolQ/WSC/WiC, 5 seeds; SAM rho grid {1e-2, 5e-3, 1e-3}):

  | Model | Adam | Adam+SAM | Delta |
  |---|---|---|---|
  | BERT-large | 79.35 | 79.85 | +0.50 |
  | ELECTRA-large | 85.68 | 85.61 | -0.07 |
  | ALBERT-xxlarge | 86.49 | 86.50 | +0.01 |
  | RoBERTa-large | 85.15 | 85.89 | +0.74 |

  In low-resource subsampling (10% to 90%) they report "consistent gains from both SAM and our FSAM across all sizes of sub-sampled training sets". - [FSAM22](https://aclanthology.org/2022.findings-emnlp.300.pdf)
- **RoBERTa-base GLUE dev, deltas vs Adam baseline (VERIFIED).** Kaddour22 Fig. 4a:
  - CoLA +1.57±1.20
  - SST −0.23±0.40
  - MRPC +0.73±0.43
  - STSB +0.38±0.17
  - QQP +0.08±0.07
  - MNLI +0.39±0.02
  - QNLI +0.09±0.01
  - RTE +0.70±0.65

  Text: "SAM achieves the best performance in 7/10 experiments on NLP tasks, consistent with the findings of Bahri et al." - [Kaddour22](https://arxiv.org/pdf/2202.00661)
- **Fine-tuning a converged model with SAM (VERIFIED).** AF22 abstract: "fine-tuning a standard model with SAM can lead to significant generalization improvements." Body: "enabling SAM only towards the end of training is sufficient to get a significant improvement in terms of generalization." This is within one task, not cross-task transfer. - [AF22](https://arxiv.org/pdf/2206.06232)
- **SAM helps more with less data, ViT setting (PARTIAL for this specific claim).** Bahri22 cites it: "Prior work (Chen et al., 2021) showed that for vision models and tasks, SAM helps more when there is less training data to learn from." I verified only the citing sentence in Bahri22, not the original data-size experiment in Chen22. - [Bahri22](https://arxiv.org/pdf/2110.08529)
- **Code models / SE / vulnerability detection: none found.**
  - Searches run: WebSearch "sharpness-aware" + CodeBERT/GraphCodeBERT/CodeT5; WebSearch "sharpness-aware minimization" + vulnerability detection/defect prediction/code smell; arXiv API abstract searches for "sharpness-aware" AND {vulnerability, "source code", code in cs.SE, "code generation"/"program repair"/"code search"/"code summarization", "programming language"}; arXiv API "flat minima" AND code in cs.SE.
  - Result: no paper training a vulnerability detector or SE classifier with SAM/ASAM.
  - The only SAM + code-model paper found is STAB26 (PARTIAL, abstract only). It trains "a surrogate model using Sharpness-Aware Minimization" to craft transferable backdoor triggers. It is motivated by "the observation that adversarial perturbations in flat regions of the loss landscape transfer more effectively across datasets than those in sharp minima". - [STAB26 abstract via arXiv API](https://export.arxiv.org/api/query?id_list=2602.11213)

### Inferences
- **Fit: moderate for "ASAM can help a small target set", weak as a full explanation.** 456 training functions is in the regime where Bahri22 saw its largest relative gains. For scale, 5% of SuperGLUE tasks is tens to a few thousand examples. So P1 is plausible under this literature.
- This explanation alone predicts a gain **regardless of Phase 1**, so it fits P2 and P3 poorly. With no Phase 1 (P2), the effect was +0.030 in one block and +0.001 in another. With C/C++ checkpoints (P3) it was mixed. P4 (ASAM making transfer itself more useful) is not predicted by a pure "small-data" story.
- The multi-seed encoder evidence matters here. On BERT/RoBERTa/ELECTRA/ALBERT, vanilla SAM's GLUE/SuperGLUE average gains are between −0.07 and +0.74 points (FSAM22), and per-task deltas often sit inside one standard deviation (Kaddour22). That is the same order as MultiVD's own noise floors (0.010 rerun, 0.028 cross-GPU per CLAUDE.md §2). Bahri22's larger gains come from single-seed, best-checkpoint-per-metric reporting on T5. This supports the project's rule that n=5 on one machine is not enough before claiming an ASAM effect.
- A paper using ASAM for vulnerability detection would be a first in the SE literature, as far as these searches can tell. It should be phrased as "we found no prior use", not "there is none".

### Gaps
- No found paper reports SAM/ASAM fine-tuning results specifically for RoBERTa-sized encoders at a few hundred training examples, with multiple seeds and ROC-AUC.
- I did not open Chen22's data-size ablation directly. The "SAM helps more with less data" claim for vision is relayed via Bahri22.
- No literature found on SAM/ASAM combined with or without LR warmup. This is relevant to P2's +0.001 block, which had "no LR warmup".

---

## Q3. SAM and label noise (Foret et al.; Baek, Kolter, Raghunathan)

### Takeaway
SAM's largest documented gains are under heavy synthetic label noise **in the training set it optimizes**: tens of points at 40-80% symmetric noise on CIFAR-10. The mechanism, per Baek24, is a trajectory effect that matters at early stopping, mainly via an implicit ℓ2 penalty on last-layer weights and activations. MultiVD applies ASAM only on the cleaner target set (~6% noise), not on the noisy source. So label-noise robustness explains little of the measured pattern directly.

### Cited Findings
- **Foret21 (VERIFIED).** "SAM natively provides robustness to label noise on par with that provided by state-of-the-art procedures that specifically target learning with noisy labels."
  - Table 4: ResNet-32, CIFAR-10, symmetric noise, clean test set, 200 epochs.

    | Noise rate | 20% | 40% | 60% | 80% |
    |---|---|---|---|---|
    | SGD | 84.8 | 68.8 | 48.2 | 26.2 |
    | SAM | 95.1 | 93.4 | 90.5 | 77.9 |
    | Bootstrap + SAM | 95.4 | 94.2 | 91.8 | 79.9 |

  - Settings: "we use ρ = 0.1 for all noise levels except 80%, for which we use ρ = 0.05 for more stable convergence." - [Foret21](https://arxiv.org/pdf/2010.01412)
- **Kwon21, ASAM under label noise (VERIFIED).** Table 6: ResNet-32 on CIFAR-10, 3 runs, max test accuracy.

  | Noise | SGD | SAM | ASAM |
  |---|---|---|---|
  | 0% | 94.50±0.11 | 94.80±0.12 | 94.88±0.12 |
  | 20% | 91.32±0.23 | 92.94±0.12 | 93.21±0.10 |
  | 40% | 87.68 | 90.62 | 90.89 |
  | 60% | 82.50 | 86.58 | 87.41 |
  | 80% | 68.35±0.85 | 69.92±0.98 | 67.69±1.34 |

  At 80% noise ASAM is below both SAM and SGD. Text: "ASAM generally enhances the test accuracy across various noise level by retaining the robustness to label noise." - [Kwon21](https://arxiv.org/pdf/2102.11600)
- **Baek24 mechanism (VERIFIED).**
  - "the peak performance under label noise occurs with early stopping, far before the loss converges."
  - "We decompose SAM's robustness into two effects: one induced by changes to the logit term and the other induced by changes to the network Jacobian. The first can be observed in linear logistic regression where SAM provably up-weights the gradient contribution from clean examples."
  - "when we intervene and modify SAM to remove this effect, surprisingly, we see no visible degradation in performance. We infer that SAM's effect in deeper networks is instead explained entirely by the effect SAM has on the network Jacobian."
  - For 2-layer linear networks, J-SAM "decomposes into SGD with ℓ2 regularization on the final layer weights and intermediate activations ... it keeps the loss of correctly fit points high by constraining the magnitude of the network output."
  - "SAM's label noise robustness may not come via sharpness properties at convergence, but rather from the optimization trajectory taken."
  - Their CIFAR-10 30% noise example: "SAM's best test accuracy is 17% higher." - [Baek24](https://arxiv.org/pdf/2405.03676)
- **SAM eventually fits noise too (VERIFIED).** AF22: at 60% label noise "SAM noticeably improves generalization over ERM, although later in training SAM also starts to fit the noisy points ... Thus, SAM also requires early stopping either explicitly via a validation set or implicitly via restricting the number of training epochs." - [AF22](https://arxiv.org/pdf/2206.06232)
- **Fine-tuning under noisy labels (VERIFIED, abstract/intro).** Ju22: overfitting in fine-tuning "has often been observed (e.g., when the target dataset is small or when the training labels are noisy)". They derive Hessian-distance bounds that also cover class-conditional label noise. - [Ju22](https://arxiv.org/pdf/2206.02659)

### Inferences
- **Fit: weak.** All noise-robustness results apply SAM to the noisy data being fit.
  - In MultiVD the noisy data (CleanVul/PrimeVul) is fit in Phase 1 **without** SAM.
  - ASAM sees only SVEN, which is about 6% noise.
  - Kwon21's table brackets the expected gain at low noise: about +0.4 pp at 0% noise and +1.9 pp at 20% noise (ASAM vs SGD, CIFAR-10 accuracy).
  - So a "noise-robustness" story predicts at most a modest, source-independent Phase-2 effect. It does not predict P3 or P4.
- **One indirect route remains (speculative).** Baek24's mechanism is an implicit penalty on last-layer weight and activation norms that slows fitting of hard (mislabeled or atypical) points. In Phase 2 this could also slow fitting of the ~6% wrong SVEN labels and of near-duplicate pairs with opposite labels in the random split (a deliberate feature of `norm`, CLAUDE.md §6). That would help ranking more than a hard threshold. This is untested.
- The literature points to a cleaner experiment for the noise story: ASAM in **Phase 1** on the noisy source. This matches the local note's "SAM may be at the wrong phase" (RESEARCH_2026-08-20_0959.md, line 207). Per CLAUDE.md §9, this would need user approval before being queued.

### Gaps
- No published measurement of SAM/ASAM at realistic low noise rates (5-10%) on small fine-tuning sets with ROC-AUC.
- No evidence on whether SAM in a *later* fine-tuning stage undoes noise memorized in an *earlier* stage.

---

## Q4. SAM and feature quality: does SAM make features (from pretraining/Phase 1) more reusable?

### Takeaway
Springer24 shows that SAM learns a more balanced, more linearly decodable set of features and transfers better under linear probing (DomainBed), by suppressing already-learned features and re-weighting poorly fit examples. Andr23LR shows SAM lowers feature rank, but low rank by itself does not explain better generalization. Both results concern SAM in the stage that *learns* the representation. No paper found shows that SAM in a *downstream* fine-tune preserves or better reuses features learned upstream.

### Cited Findings
- **Springer24 abstract (VERIFIED).**
  - "we argue that SAM implicitly balances the quality of diverse features. SAM achieves this effect by adaptively suppressing well-learned features which gives remaining features opportunity to be learned."
  - "this mechanism is beneficial in datasets that contain redundant or spurious features where SGD falls for the simplicity bias and would not otherwise learn all available features."
  - Two effects: "a balancing of the weight of the training examples during each update step, leading to a more uniform update across the dataset", and "the effective learning rates of well-learned features are suppressed."
  - "This bias makes SAM favorable in downstream tasks with distributed shifts where a diverse set of features become relevant for prediction."
  - DomainBed numbers are in Q1. - [Springer24](https://arxiv.org/pdf/2405.20439)
- **Andr23LR (VERIFIED).** "we uncover an additional intriguing effect of SAM: reduction of the feature rank which happens at different layers of a neural network", observed "for different architectures such as fully-connected networks, convolutional networks, vision transformers". But: "we show that directly inducing low-rank features does not improve generalization on natural data. Thus, we conclude that SAM is unique in its ability to both improve generalization and decrease the feature rank in a data-adaptive fashion." - [Andr23LR](https://arxiv.org/pdf/2305.16292)
- **SAM-learned upstream features transfer better even with non-SAM fine-tuning (VERIFIED).** Chen22 Table 10 (numbers in Q1). - [Chen22](https://arxiv.org/pdf/2106.01548)
- **Function-space view (VERIFIED).** TRAM24 argues that "generalization during fine-tuning is often more dependent on the transferability of representations in the function space", and that trust-region methods "reduce catastrophic forgetting of pretrained task-agnostic information". TRAM, which adds a representation trust region to ASAM, beats ASAM on XNLI zero-shot (75.0-75.2 vs 74.0). - [TRAM24](https://arxiv.org/pdf/2310.03646)

### Inferences
- **Fit: moderate as a mechanism hypothesis for P1/P3/P4; not directly evidenced.** Springer24's balancing effect runs during whichever phase uses SAM. Applied in Phase 2, it predicts the following:
  - ASAM should stop the target fine-tune from collapsing onto the few "easy" features most salient in the Phase-1 representation.
  - It should keep updating the parts of the representation that fit the small target set poorly.
- Whether that helps should depend on **which** features Phase 1 left behind. This dependence is consistent with P1 (JS works) vs P3 (C/C++ mixed). It would also produce an interaction (P4): balancing only pays off when the starting representation contains several useful but unequally learned features.
- A better feature ordering improves ranking (ROC/PR-AUC) without necessarily fixing the operating point at 0.5, consistent with P5.
- All of this is inference. The cited experiments put SAM in the representation-learning stage and measure with linear probes. A cheap diagnostic in the spirit of Springer24 is to linear-probe frozen Phase-2 representations (AdamW vs ASAM, same fold) on held-out SVEN data. This would test the mechanism without new training runs, since it reuses existing checkpoints if they were saved.
- Low feature rank (Andr23LR) is not an explanation by itself: the authors show that inducing low rank directly does not improve generalization.

### Gaps
- No paper found that applies SAM only in a second-stage fine-tune and measures feature diversity or reuse of first-stage features.
- No code-domain evidence on SAM and feature diversity.

---

## Q5. SAM and forgetting / staying near pretrained weights

### Takeaway
Strong recent evidence says a **flat starting point** forgets less when later fine-tuned, and that forgetting scales with directional sharpness times squared distance moved:
- Watts26: SAM in pretraining or mid-training gives up to 80% less forgetting.
- Mehta23: pretraining gives flatter task minima, and SAM during sequential fine-tuning reduces forgetting.

Two further results give a theoretical bridge to the P4 interaction:
- Ju22: fine-tuning generalization is bounded by a Hessian-weighted distance from the initialization.
- AF22: fine-tuning with SAM stays in the starting basin.

In MultiVD, however, the *starting point* (Phase-1 checkpoint) was trained without SAM. So the best-evidenced version of this explanation (flat Phase 1) has not been tested in the project.

### Cited Findings
- **Watts26 (VERIFIED, ICML 2026 per arXiv comment; preprint read).**
  - Abstract: interventions that "bias optimization toward flatter minima: Sharpness-Aware Minimization (SAM), large learning rates, and shortened learning rate annealing periods. Across model sizes ranging from 20M to 150M parameters, we find that these interventions consistently improve downstream performance after post-training on five common datasets with up to 80% less forgetting. ... a short SAM mid-training phase applied to an existing OLMo-2-1B checkpoint reduces forgetting by 31% after MetaMath post-training and by 40% after 4-bit quantization."
  - Code is one of the post-training sets: "SAM produces 80% less forgetting on StarCoder at matched fine-tuning loss".
  - Mechanism: "The degradation of the loss after fine-tuning is determined by the product of the directional sharpness and the (squared) distance traveled during fine-tuning. This implies that the reduced sensitivity induced by SAM, in fact, arises from the flattening of the base model along the direction of fine-tuning, rather than from any reduction in the fine-tuning step size."
  - Limitation: at 1B, SAM mid-training "did not improve forgetting after post-training on MusicPile".
  - SAM rho in mid-training: "ρ = 0.05".
  - Forgetting here means loss on the *pretraining* evaluation after fine-tuning. A grep of the PDF found no experiment that applies SAM during the post-training stage itself. - [Watts26](https://arxiv.org/pdf/2605.02105)
- **Mehta23 (VERIFIED).**
  - Abstract: "pre-trained weights appear to ease forgetting by leading to wider minima. Based on this insight, we propose jointly optimizing for current task loss and loss basin sharpness to explicitly encourage wider basins during sequential fine-tuning."
  - From their Eq. 2 (forgetting is about ½·λmax·‖Δw‖²): "the flatter the minima, the less forgetting occurs in the model."
  - Table 4, DistilBERT-PT, FT vs FT w/ SAM (mean with std over 5 runs):

    | Benchmark | Accuracy | Forgetting |
    |---|---|---|
    | 5-dataset-NLP | 64.3±4.5 to 66.4±2.8 | 16.7±5.7 to 13.9±3.5 |
    | Split YahooQA | 87.7 to 88.5 | 9.5±4.7 to 8.4±3.5 |

    Vision effects are larger. ResNet-18-PT on 5-dataset-CV: accuracy 57.2 to 70.4, forgetting 38.3 to 25.6.
  - "SAM results in a consistent improvement in performance over non-SAM counterparts." - [Mehta23](https://arxiv.org/pdf/2112.09153)
- **Ju22 (VERIFIED).**
  - "Our motivating observation is that in addition to distance from initialization, Hessian crucially affects generalization."
  - The generalization-error bound takes the form Σ_i sqrt(max over (x,y) of v_iᵀ H_i⁺ v_i), divided by sqrt(n). Here v_i is the flattened W_i − W_i^(s), the displacement of layer i from the pretrained initialization, and H_i is the loss Hessian over W_i.
  - "incorporating Hessian into the distance-based measure accurately correlates with the generalization error of fine-tuning." - [Ju22](https://arxiv.org/pdf/2206.02659)
- **AF22 (VERIFIED).** Switching ERM to SAM at the end of training: "the test error clearly improves when switching from ERM to SAM ... SAM (using a higher ρ than the standard value ...) can gradually escape the worse-generalizing minimum which ERM converged to. ... we can start from any pre-trained model and substantially improve its generalization. Moreover, interestingly, the final point of the ERM → SAM model is situated in the same basin as the original ERM model." - [AF22](https://arxiv.org/pdf/2206.06232)
- **Sadr23 (PARTIAL, abstract).** Models fine-tuned from one checkpoint "end up in the same basin of the loss landscape, which we call the pre-train basin ... leaving the basin results in losing the benefits of transfer learning". - [Sadr23](https://arxiv.org/pdf/2303.03374)

### Inferences
- **Fit: best available account of the P4 interaction, but only as a hypothesis.** Ju22 gives a bound that is a product: curvature along the displacement, times the displacement from the starting checkpoint, divided by sqrt(n). With n=456, this term is large.
  - A useful Phase-1 checkpoint can shorten the displacement needed on the target (smaller v).
  - ASAM penalizes curvature along the path (smaller H along v).
  - Together they could lower the bound multiplicatively rather than additively. That matches "transfer helps +0.016 with AdamW but +0.043 with ASAM".
  - AF22's "same basin" result is consistent with SAM polishing *within* the Phase-1 basin rather than leaving it. Sadr23 says leaving it loses the transfer benefit.
  - The size of this effect is not derivable from these papers.
- **Mismatch with the strongest evidence.** Watts26 and Mehta23 locate the benefit in the flatness of the **earlier** solution (θ_PT, or task-1 minimum in Mehta's Eq. 2). MultiVD's Phase 1 is trained without SAM. So these papers argue more for testing ASAM in Phase 1, which per CLAUDE.md §9 must be proposed to the user, not queued. They do not by themselves explain why Phase-2-only ASAM helps.
- **Dataset dependence fits P3.** Watts26 reports no benefit on one of four post-training sets (MusicPile). Mehta23's NLP effects are within about one standard deviation. Benefits of flatness for transfer are dataset-dependent, consistent with JS vs C/C++ checkpoints behaving differently.
- **Measurable without new training, if checkpoints exist.** Two quantities, per fold and per checkpoint source:
  1. distance ‖θ_P2 − θ_P1‖ for AdamW vs ASAM;
  2. directional sharpness of the target loss along θ_P2 − θ_P1 (Watts26's quantity).

  The P4 interaction story predicts that ASAM reduces directional sharpness more (or travels less) when starting from the JS checkpoint than from C/C++ checkpoints.

### Gaps
- No paper found isolates **SAM in the second stage only** (non-SAM first stage) and measures retention of first-stage features. Watts26 explicitly lists "adapters, and alternatives to SFT" and other regimes as open. I found no SAM-at-post-training ablation in their text.
- Mehta23 NLP effects are small relative to their stds. No code-model lifelong-learning SAM study found.

---

## Q6. SAM and calibration vs ranking: does SAM improve calibration/confidence or ranking (AUC) rather than accuracy at a fixed threshold?

### Takeaway
Tan26 (ICLR 2026) shows that SAM mainly fixes **overconfidence**, by implicitly maximizing predictive entropy. It roughly halves ECE while moving accuracy little. Baek24 shows SAM constrains output magnitude. I found **no** paper that reports SAM improving classification ROC-AUC more consistently than accuracy or F1 at a fixed threshold. The AUROC numbers in Tan26 are **OOD-detection** AUROC, not in-distribution ranking.

### Cited Findings
- **Tan26 (VERIFIED).**
  - Abstract: "we show that the recently proposed sharpness-aware minimization (SAM) counteracts this tendency towards overconfidence. The theoretical analysis suggests that SAM allows us to learn models that are already well-calibrated by implicitly maximizing the entropy of the predictive distribution."
  - Contribution: SAM "performs an implicit regularization on the negative entropy of the predictive distribution. This is similar to focal loss ... but SAM calibrates models much better without compromising accuracy."
  - Table 1 (ResNet-18, CIFAR-10, vanilla): SGD test accuracy 89.18±0.26, ECE 5.76±0.43; SAM accuracy 90.01±0.23, ECE 3.24±0.39.
  - Text: "ECE of SGD approximately remains two times larger than ECE of SAM. This is different from their behavior on test accuracy".
  - The "OOD AUROC" columns (SVHN, CIFAR10-C, CIFAR100-C) measure OOD detection: SGD 83.94 vs SAM 86.38 on SVHN. They are not binary-classification ranking. - [Tan26](https://arxiv.org/pdf/2505.23866)
- **Baek24 (VERIFIED).** J-SAM acts as "ℓ2 regularization on the final layer weights and intermediate activations", which keeps "the loss of correctly fit points high by constraining the magnitude of the network output." - [Baek24](https://arxiv.org/pdf/2405.03676)
- **Kaddour22 (VERIFIED).** Uses ROC-AUC on OGB-Proteins, where "no flat optimizer improves over the baseline optimizer". This is the only ROC-AUC-scored SAM result I encountered, and it is a null. - [Kaddour22](https://arxiv.org/pdf/2202.00661)

### Inferences
- **The calibration story does not explain an AUC gain by itself.** ROC-AUC and PR-AUC are invariant to any strictly monotone transform of the score. A pure "less overconfident" effect (temperature-like shrinking of logits, Tan26/Baek24) changes ECE but leaves AUC unchanged. In binary classification, pure logit scaling also leaves the 0.5 decision unchanged, since the sign of the logit is unchanged. So the ASAM gain in AUC (P5) must come from changed **ordering** of examples, i.e. representation or decision-function changes (Q4/Q5), not from calibration.
- **Why F1@0.5 tracks less well (inference).** A change in ordering improves AUC everywhere on the curve. F1@0.5 depends on where the decision bias lands on a 152-example test fold, which is a noisier statistic. Any fold-to-fold shift in the effective bias (from training dynamics or from SAM's output-norm shrinkage moving borderline examples) adds variance to F1@0.5 but not to AUC. This is consistent with P5's 119/190 (ROC) vs 100/190 (F1@0.5) sign counts.
- **Check that separates "ranking" from "threshold".** CLAUDE.md §2b already requires F1 at the val-tuned threshold. If ASAM's gain shows up in AUC and in F1@val-threshold but not in F1@0.5, the effect is ordering plus miscalibrated bias. If it shows up in none of the F1 variants, the AUC gain is in a region of the ROC curve away from either operating point.
- **Prediction from Tan26.** Measure ECE or Brier score on the existing per-sample test probabilities (the project logs them, per memory). Tan26 predicts ECE drops with ASAM. Whether that coincides with the AUC gain is an open empirical question for the project.

### Gaps
- No paper found that reports in-distribution ROC-AUC vs accuracy/F1 for SAM/ASAM in binary classification, especially for imbalanced or security detection.
- Tan26's experiments are vision models trained from scratch, plus some fine-tuning. I did not find language-model fine-tuning calibration numbers in the parts I read.

---

## Q7. ASAM specifics: recommended rho and eta, and typical rho for transformer fine-tuning

### Takeaway
- **ASAM paper (Kwon21):**
  - SGD from scratch: rho = 0.5 (CIFAR-10) or 1.0 (CIFAR-100, ImageNet).
  - Transformer on IWSLT'14 with Adam: rho = 0.2.
  - eta = 0.01.
- **ASAM in later transformer work:** TRAM24 fine-tunes XLM-R Base with Adam at rho = 0.5.

MultiVD's rho = 0.5, eta = 0.01 is therefore within published practice. It matches TRAM24's transformer fine-tuning value and is 2.5x Kwon21's transformer value. Non-adaptive SAM values for LMs (0.05-0.15 for T5; 1e-3 to 1e-2 for BERT-family in FSAM22) are not comparable, because ASAM's rho is relative to |w|.

### Cited Findings
- **Kwon21 definition (VERIFIED).** "we use Tw + ηIk rather than Tw for sufficiently small η > 0 for stability. η is a hyper-parameter controlling trade-off between adaptivity and stability." With element-wise normalization Tw = diag(|w1|, ..., |wk|), each parameter's perturbation radius scales with |w_i| + η. Other choices:
  - "we decide to use element-wise normalization operator and p = 2, and not to employ bias normalization"
  - "η for ASAM is set to 0.01" - [Kwon21](https://arxiv.org/pdf/2102.11600)
- **Kwon21, CIFAR (VERIFIED).** "we first conduct a grid search over {0.00005, 0.0001, 0.0002, . . . , 0.5, 1.0, 2.0} for finding appropriate values of ρ. We use ρ = 0.5 for CIFAR-10 and ρ = 1.0 for CIFAR-100, because it gives moderately good performance across various models. We set ρ for SAM as 0.05 for CIFAR-10 and 0.1 for CIFAR-100". The ImageNet run uses "ρ = 0.05 for SAM and ρ = 1.0 for ASAM". - [Kwon21](https://arxiv.org/pdf/2102.11600)
- **Kwon21, Transformer with Adam (VERIFIED).** "We choose ρ = 0.1 for SAM and ρ = 0.2 for ASAM as a result of a grid search over {0.005, 0.01, 0.02, . . . , 0.5, 1.0, 2.0} using validation dataset." Settings: Adam lr 0.0005, β=(0.9, 0.98), dropout 0.3, weight decay 0.0001, label smoothing 0.1, 3 runs. Table 5 BLEU on IWSLT'14 DE-EN:

  | | Adam | Adam+SAM | Adam+ASAM |
  |---|---|---|---|
  | Validation | 35.34 | 35.52 | 35.66 |
  | Test | 34.86 | 34.78 | 35.02 |

  [Kwon21](https://arxiv.org/pdf/2102.11600)
- **Kwon21 on scale invariance (VERIFIED).** "appropriate ρ for SAM is dependent on the scales of w on the training trajectory, whereas ρ of ASAM is not." - [Kwon21](https://arxiv.org/pdf/2102.11600)
- **ASAM in transformer fine-tuning (VERIFIED).** TRAM24 Appendix: "we compare to SAM (ρ = 0.05, Foret et al., 2021), Adaptive SAM (ASAM, ρ = 0.5, Kwon et al., 2021)". All language methods "use Adam as the inner optimizer". The model is "the 250M XLM-Roberta Base multilingual" model. Results are in Q1. TRAM24 also chose ASAM over SAM as its base "after observing strictly better performance". - [TRAM24](https://arxiv.org/pdf/2310.03646)
- **Non-adaptive SAM values for LM fine-tuning (VERIFIED).**
  - Bahri22: T5 rho 0.05 (Small) and 0.15 (Base and larger); mT5 needed smaller ({0.01, 0.02, 0.05}). Sensitivity: "all tested values perform better than fine-tuning without SAM. However, 0.15 is a 'sweet spot'". - [Bahri22](https://arxiv.org/pdf/2110.08529)
  - FSAM22: SAM grid {1e-2, 5e-3, 1e-3} for BERT/RoBERTa/ELECTRA/ALBERT. - [FSAM22](https://aclanthology.org/2022.findings-emnlp.300.pdf)
- **Larger rho for SAM fine-tuning of a converged model (VERIFIED).** AF22's ERM-to-SAM switch uses "a higher ρ than the standard value". - [AF22](https://arxiv.org/pdf/2206.06232)
- **Local project record (local file, not a publication).** ASAM rho in {0.1, 0.2, 0.5} in Phase 2 on CodeT5+ (t5p) was "not above the noise floor at n=15" (FACTS §18, as summarized in `/drive1/cuongtm/ntat/MultiVD/RESEARCH_2026-09-06_recadam.md` line 91).

### Inferences
- **Fit:** rho = 0.5, eta = 0.01 is a defensible published setting. It is Kwon21's CIFAR-10 value and TRAM24's XLM-R fine-tuning value. The only transformer value tuned in the ASAM paper itself is 0.2, for training from scratch on IWSLT.
- Since the effective perturbation is ρ·(|w|+η) per coordinate, the actual step differs between a CodeBERT checkpoint and a Phase-1 checkpoint only through their weight magnitudes. ASAM's scale invariance is meant to make one rho transfer across starting points. The source-dependence in P3 is therefore unlikely to be a rho-scale artifact, though this was not measured.
- The CodeT5+ null at n=15 (local FACTS §18) vs the CodeBERT+JS positive at n=5 is consistent with Kaddour22's "architectures matter". Under CLAUDE.md §2 (n=5 on one machine is not enough), the CodeBERT result still needs a replicate on other hardware before it is reported as a finding.

### Gaps
- No published ASAM rho sweep for BERT/RoBERTa-sized encoders fine-tuned on a few hundred examples.
- No published guidance on ASAM combined with LR warmup or schedule (relevant to P2's no-warmup block).
- I did not read Kwon21's appendix figures on rho sensitivity (Figure 4 is described in text only).
