# Nghiên cứu sâu — 08/09/2026 — Làm sao biến phần TRANSFER thành đóng góp

> Chạy bằng `/deep-research`: 105 agent, 5 hướng tìm kiếm, xác minh đối kháng 3 phiếu mỗi
> luận điểm (cần 2/3 bác mới loại). Tổng 1 685 lượt gọi công cụ.

## Tóm tắt của báo cáo

The verified evidence converges on one negative and three positive conclusions. Negative: none
of the weight-space transfer machinery is a good bet at MultiVD's scale — mAdapter's cross-
language adapter composition gains only +0.18 BLEU-4 / +0.8 MRR at CodeT5-base 220M and
UniXcoder 126M (and goes *below* the monolingual baseline on Go and PHP, p=0.986/0.445), He et
al. measure only +0.7%/+0.4% on the stronger backbone at 1k/5k GLUE, MAD-X's headline >5 F1
comes from a target-language MLM module (a published null at high-resource targets: XQuAD 70.3
vs 70.6), and TIES-Merging never reaches individual fine-tuning in its own headline table
(T5-Base 82.8 vs 73.9) — all at or below the 0.010 macro-F1 noise floor and all confirming
MultiVD's existing WiSE-FT null. Positive: three contribution shapes survive verification as
established, citable, and matched to what MultiVD has *already* measured — (1) intermediate-
task/source selection, a named problem with a standard protocol (NDCG + regret@k over a large
source pool) validated at ~110-167M encoders with 1k-row targets; (2) negative-transfer
characterization, quantified at ACL main track with exactly MultiVD's asymmetry (best source
+2.6, worst -9.9 avg, -31.0 tail, driven by degenerate runs on small targets — the same
mechanism as MultiVD's val=0.3403 collapse and -0.4395 delta); and (3) AdvFusion's (SANER 2025)
"measure that the assumed mechanism does not fire, then design an intervention that makes it
fire", which is the shape MultiVD's four null results already half-occupy with a stronger
diagnostic package than the single attention figure AdvFusion published. Critically, the
verification pass returned zero surviving claims on three of the five requested angles —
function-space/distillation transfer, joint/rehearsal/gradient-surgery training, and
SE/security-venue evaluation protocols — so this synthesis cannot rank those, and their absence
reflects search coverage, not evidence of absence.

## Các luận điểm sống sót xác minh

### [1] HIGH · phiếu 3-0 on both constituent claims (merged)

Adapter-based cross-language transfer IS validated at MultiVD's exact backbone scale, but its
published effect sizes sit at or below the project's 0.010 noise floor and reverse per-language
— so swapping sequential fine-tuning for adapters is a hyperparameter-class change here, not a
contribution.

- https://arxiv.org/abs/2303.15822
- https://aclanthology.org/2021.acl-long.172
- https://arxiv.org/abs/2106.03164

### [2] HIGH · phiếu 3-0 (mechanism/attribution) and 2-1 (XQuAD null); all numbers verified exactly against the primary PDF

MAD-X-style language-adapter composition is the wrong analogue for MultiVD: its >5 F1 headline
comes overwhelmingly from a language adapter trained by MLM on unlabelled TARGET-language data,
it is concentrated on pretraining-unseen low-resource languages, and it is a published null when
the target language is high-resource. Python is thoroughly covered by every code backbone's
pretraining and MultiVD has 456 labelled target rows, so the mechanism that produces the gain is
absent.

- https://arxiv.org/pdf/2005.00052
- https://aclanthology.org/2020.emnlp-main.617.pdf

### [3] HIGH · phiếu 3-0

LT-SFT is the strongest citable precedent that COMPOSING separately-trained deltas by plain
vector addition is a defensible methodological contribution distinct from sequential fine-tuning
— validated at 110-270M encoders, pre-dating TIES/DARE. But its advantage depends on a target-
language MLM module MultiVD does not have, and it never runs head-to-head against sequential
full fine-tuning.

- https://aclanthology.org/2022.acl-long.125.pdf
- https://arxiv.org/abs/2110.07560

### [4] HIGH · phiếu 3-0

Weight-space merging (task arithmetic, TIES, DARE, soups) should not be pursued: in TIES-
Merging's own headline table no merging method ever reaches individual fine-tuning or joint
multitask training at 220M scale, and the authors name 'no established method for selecting
WHICH checkpoints to merge' as an open problem. This independently corroborates MultiVD's local
WiSE-FT null (optimal alpha = 1.0 in 10/10 cells).

- https://papers.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf
- https://arxiv.org/abs/2306.01708

### [5] HIGH · phiếu 3-0 on both constituent claims (merged; same primary paper)

'Which source do I pick' is an established, named research problem — intermediate task selection
— with a standing line of citable methods and a standard evaluation protocol (NDCG plus
regret@k, k=1,3,5), validated at ~110-167M encoder scale with target sets deliberately truncated
to 1k rows and Phase-2 hyperparameters of 3 epochs at lr 2e-5. This is the closest published
regime match to MultiVD. BUT the recognized contribution shape is building a transferability
ESTIMATOR that RANKS a large source pool, not hand-picking one filtered source and reporting
that it works.

- https://aclanthology.org/2024.emnlp-main.529/
- https://arxiv.org/abs/2410.15148
- https://arxiv.org/abs/2104.08247

### [6] HIGH · phiếu 3-0

Severe, systematic negative transfer from a poorly-matched source is a quantified, peer-reviewed
phenomenon at encoder scale, and the asymmetry it documents (small upside, very large downside)
makes negative-transfer characterization an accepted contribution shape in its own right.
MultiVD's -0.4395 macro-F1 unfiltered-source collapse is in-family with published behaviour —
including the mechanism.

- https://aclanthology.org/2020.acl-main.467/

### [7] HIGH · phiếu 3-0 on the claim; payoff evidence is mixed

The contribution shape most directly available to MultiVD is 'measure that the assumed transfer
mechanism does not fire, then design an intervention that makes it fire' — AdvFusion is a
citable precedent at CodeBERT 125M / CodeT5+ 220M. MultiVD already holds a stronger diagnostic
package than the one AdvFusion published; what is missing is the intervention half.

- https://arxiv.org/abs/2307.07854

### [8] MEDIUM · phiếu None

PRIORITIZED SHORTLIST (synthesis judgement, not a single verified claim). Rank order: 1) source
selection reframed as a transferability estimator; 2) negative-transfer characterization; 3)
diagnose-then-intervene on the non-firing mechanism; 4) sparse composable delta composition (LT-
SFT); 5) adapters as a drop-in; 6) do not pursue weight-space merging or MAD-X language
adapters.

- https://aclanthology.org/2024.emnlp-main.529/
- https://aclanthology.org/2020.acl-main.467/
- https://arxiv.org/abs/2307.07854
- https://aclanthology.org/2022.acl-long.125.pdf
- https://arxiv.org/abs/2303.15822
- https://arxiv.org/abs/2106.03164
- https://papers.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf
- https://arxiv.org/pdf/2005.00052

## Luận điểm BỊ BÁC (2/3 phiếu trở lên) — đọc kỹ, vài cái là cảnh báo cho chính ta

- Adapter tuning beats full-model fine-tuning ONLY in cross-lingual settings; on monolingual
tasks (train and test in the same language) full fine-tuning wins, because more parameters can
be adjusted to fit the task. This is a direct scope limit for MultiVD: Phase 2 is a same-
language (Python) fit, so an adapter would be expected to underperform full fine-tuning there —
the paper's own diagonal cells are the counter-evidence.

- Joint multilingual full-model fine-tuning (mixing source-language data into training, i.e. the
rehearsal/joint-training angle) is backbone-dependent and NEGATIVE on the two modern backbones
MultiVD uses: it helps CodeBERT/GraphCodeBERT on all languages, but degrades UniXcoder and
CodeT5 on all but low-resource languages. Concretely mCodeT5 (232M, multilingual full FT) scores
19.41 overall BLEU-4 vs 19.56 for per-language monolingual CodeT5, and mUniXcoder 74.6 MRR vs
74.5 for monolingual UniXcoder on code search.

- Sparsity is causally necessary for composability: sweeping task-SFT density x language-SFT
density from 5% to 100%, performance degrades markedly once density exceeds ~30%, which the
authors attribute to interference between the composed fine-tunings. The authors explicitly flag
that overfitting from excess capacity is an unexcluded alternative explanation. This means naive
dense delta-addition (i.e. full-parameter task arithmetic) is predicted to FAIL, and any
composition scheme must enforce sparsity/disjointness.

- Adapter composition is a NULL result at CodeT5+ 220M — the exact backbone class in the MultiVD
setting. In Table III, CodeT5p+AdvFusion vs CodeT5p+AdapterFusion differ by at most 0.14 BLEU-4
and AdvFusion is WORSE on two of six languages (Ruby 14.70 vs 14.79; Go 18.25 vs 18.30; JS 14.96
vs 14.82; Python 18.98 vs 18.94; Java 18.78 vs 18.71; PHP 23.87 vs 23.80). Plain full fine-
tuning of CodeT5p beats CodeT5p+AdvFusion on 5 of 6 languages (e.g. Go 19.00 vs 18.25, Python
19.77 vs 18.98, PHP 25.13 vs 23.87), losing only on Ruby (14.55 vs 14.70). The paper's own
explanation is architectural, not data-dependent: encoder-decoder models with a pre-trained
decoder resist adapter insertion. This is direct published evidence that AdapterFusion/MAD-X-
style adapter composition should NOT be expected to beat sequential fine-tuning on a CodeT5+
backbone.

- The cross-language transfer gain is confined to low-resource targets and REVERSES on high-
resource ones — a published instance of the 'transfer gain grows as target data shrinks'
evaluation shape. With CodeBERT, AdvFusion beats full fine-tuning by +4.37 BLEU on Ruby (16.53
vs 12.16, the 24,927-example target) but LOSES to full fine-tuning on Python (18.28 vs 19.06,
the 251,820-example target); GraphCodeBERT shows the same pattern (Ruby 16.47 vs 12.62 full FT;
PHP 24.83 vs 25.45 full FT). The reported wins are 8%, 6%, 7%, 5% BLEU for Ruby, JavaScript, Go,
Java only.

- At exactly the MultiVD scale (RoBERTa-base, ~125M encoder; four tasks with 515–4169 training
examples), bottleneck adapter tuning beats full fine-tuning by +1.9% average (macro-F1 on 3 of 4
tasks), the closest published analogue to a few-hundred-row target. Per-task: HYPERPARTISAN
n=515 88.9→90.4, ACL-ARC n=1688 65.0→67.5, SCIERC n=3219 78.5→80.8, CHEMPROT n=4169 81.7→82.9,
averaged over 5 seeds. Note the two smallest tasks also carry the largest std devs (4.2 and
4.3), so single-task gains are within seed noise even though the 4-task average is not.

- The adapter advantage is a monotone function of target-set size: it is largest at the smallest
n and vanishes at high resource, where adapters and fine-tuning are indistinguishable (RCT 180k:
87.0 vs 87.1; AGNEWS 115k: 93.7 vs 93.8; HELPFULNESS 115k: 69.1 vs 69.0). Figure 3 plots this
explicitly as a transfer curve over 2k/4k/8k/16k/32k/64k/all training samples. This is a
published instance of the exact 'gain grows as target data shrinks' data-efficiency protocol,
and it predicts a NON-null effect at ~456 target rows.

- Adapters used ONLY as task adapters (i.e. as a drop-in substitute for full sequential fine-
tuning, with no language-specific module) are a null result versus full fine-tuning on languages
the model already saw in pretraining, and are actively WORSE on unseen languages: MAD-X_Base -
LAD - INV scores 29.8 avg NER F1 vs XLM-R_Base's 32.6. This means the MAD-X gain does not come
from adapter modularity per se — it comes from the target-language MLM adapter. For MultiVD,
where Python is already in every code backbone's pretraining corpus, this predicts a null from
swapping sequential FT for task adapters.

- TIES-Merging validates the merged model as a better initialization for a small-data target
recipe at ~110M encoder scale — the exact shape MultiVD would need for Phase 2 init — but the
gain is NOT universal. Merging bert-base-uncased checkpoints from the 7 other GLUE tasks and
then fine-tuning on the held-out task gives RTE 66.4 vs 66.4 PTM-init → 80.1 (+13.7), MRPC 81.8
→ 88.0 (+6.2), but WNLI 56.3 PTM-init → 54.9 TIES (−1.4, where plain Averaging merely ties PTM-
init at 56.3). Task Arithmetic also degrades WNLI (50.7). So on the smallest target (WNLI, 635
train rows) all task-vector merging methods underperform simply starting from the pretrained
model.

- TIES's own novel contribution (trim + sign election) delivers essentially nothing over the
much simpler Task Arithmetic when only 2 models are merged — both sit at ~1.0 average normalized
accuracy, i.e. negligible loss versus the individual fine-tuned models — and the TIES-over-Task-
Arithmetic gap opens only as the number of merged tasks grows toward 7-11. For a MultiVD design
merging 2-3 Phase-1 source checkpoints (ccpp/js, or 4cwe/com/full), the expected TIES-specific
delta is therefore near zero; only simple averaging is clearly penalized (~10% drop) at 2 tasks.

- Choosing the wrong intermediate task actively degrades target performance relative to no
intermediate fine-tuning at all, so the paper frames proper source selection (not the transfer
step itself) as the load-bearing decision. This is published corroboration for treating a severe
negative-transfer observation (e.g. an unfiltered source costing -0.44 macro-F1) as a legitimate
finding rather than a bug.

- Even the best intermediate-task selection method yields only marginal end-task gains:
0.38%-0.91% absolute improvement over the no-transfer baseline when selecting the single top
source task, rising to at most 1.03% when the best of the top-3 selected sources is used. The
authors themselves characterize the effectiveness of intermediate-task selection as 'still
limited'.

- Task-type / taxonomic similarity between source and target is NOT a reliable predictor of
transferability: the top-performing source tasks for a given target vary widely in task type and
task types are generally uncorrelated with transfer performance (e.g. the most beneficial
sources for COPA, a QA task, were CxC semantic-similarity and QQP paraphrase detection; many of
the best sources for CB, an NLI task, were non-NLI).

- On the smallest targets, random-seed variance swamps the source-selection signal: for COPA
(400 training examples) the same top-ranked source task produces relative improvements ranging
from 7.69% to 26.78% depending only on the training seed, and for CB (250 examples) from 4.11%
to 7.60% — variance far larger than the 0.38-1.03% absolute gain any selection method delivers.

- Intermediate-task DATA SIZE does not predict transfer benefit: in a controlled sweep varying
training data volume across five intermediate tasks (RoBERTa-Large, 355M), no consistent effect
on target performance was found. This is direct published support for the shape of the MultiVD
finding that a 930-row CWE-filtered source beats a 7,598-row unfiltered source — it licenses
'source COMPOSITION, not source SIZE' as an established, citable claim rather than an
idiosyncratic artifact.

## Cảnh báo của chính báo cáo

1) COVERAGE GAP IS THE BIGGEST CAVEAT. Three of the five requested angles produced NO claims
surviving verification: angle 3 (function-space transfer — LwF-style distillation from the
Phase-1 model on target inputs, feature matching, CKA/contrastive representation alignment),
angle 4 (joint/multi-task/rehearsal cross-language training, PCGrad and gradient-conflict
measures, language-adversarial code representations), and angle 5 (which evaluation protocols
ICSE/FSE/ISSTA/USENIX/TOSEM reviewers accept as evidence of transfer). The requested 2024-2026
survey of cross-language / cross-project vulnerability-detection transfer also returned nothing
verified. This is absent search coverage, not evidence of absence — and angle 3 is precisely the
family MultiVD's own nulls do NOT touch, since RecAdam, SPD, EWC and WiSE-FT are all weight-
space. 2) NO VERIFIED EVIDENCE IS ON VULNERABILITY DETECTION. Every surviving claim is on code
search/summarization at CodeSearchNet scale or on NLP (GLUE, NER, QA, text classification). No
verified result exists at a 456-row target; the closest are 1k rows (EMNLP 2024) and the
515-4169-row TAE tasks in He et al. — and that TAE claim was voted down 1-2. 3) REFUTATION
ASYMMETRY. Nine of the fifteen refuted claims were the pro-adapter or pro-source-selection-
effect-size ones. The verification record does not distinguish "factually false" from
"overreaching framing": e.g. the transfer-curve claim (adapter gain grows as target shrinks) is
descriptively supported inside claim [3]'s own verification notes (+2.5@1k vs +0.7@5k on BERT)
yet was refuted 0-3 as a predictive extrapolation to 456 rows. Treat every refutation as "did
not survive as stated", never as disproof — and note the net effect is that this synthesis is
biased toward pessimism about adapters. 4) CROSS-METRIC COMPARISON. Comparing BLEU-4, MRR and
GLUE-accuracy deltas against a 0.010 macro-F1 noise floor is an order-of-magnitude heuristic,
not an equivalence. The "below the noise floor" verdicts are directional, not arithmetic. 5)
TIME SENSITIVITY. TIES's "merging lags multitask training" is exact for 2023 but partly
superseded by EMR-Merging (NeurIPS 2024) and AdaMerging (ICLR 2024), which buy the gap-closing
with per-task masks, task identity at inference, or unlabelled test-time data. MAD-X (2020) is
no longer SOTA (superseded by LT-SFT and TLR task adapters). Source-selection estimators
(LEEP/LogME) have documented reliability failures out-of-domain and under seed variance
(arXiv:2407.16245), though those specific critiques did not survive verification here. 6) READ
TABLES, NOT ABSTRACTS. mAdapter's prose asserts significance "for all programming languages
except Ruby" while its own Table V prints p=0.986 (Go) and p=0.445 (PHP); Pruksachatkun's
"negative transfer on all target tasks" is 8/10 for QQP. Two of the strongest claims here are
more accurate than the papers' own summaries. 7) ATTRIBUTION SLIPS RECORDED DURING VERIFICATION,
worth fixing before citing: LT-SFT's inference-time composed module is the TARGET-language SFT
(the source side contributes the task delta) — the claim's framing inverted this; MAD-X's
32.6/38.2 are means over all 16x16 source-target pairs, not over 16 target languages; the EMNLP
2024 backbone is ~167M (mBERT), not 110M; "TaskEmb" is Vu et al.'s (2020) name for an adaptation
of Achille et al.'s Task2Vec (ICCV 2019), an imprecision inherited from the cited paper. 8)
VENUE MISMATCH. All contribution-shape precedents here are ACL/EMNLP/NeurIPS/ICSE/SANER. Only
mAdapter (ICSE 2023) and AdvFusion (SANER 2025) are SE venues, and neither is security. No
direct evidence was obtained about what SE/security reviewers accept.
