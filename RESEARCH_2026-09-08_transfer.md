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

---

# PHỤ LỤC B — LƯỚI NGUỒN: tách ĐỘ TINH KHIẾT NHÃN khỏi TỈ LỆ NGÔN NGỮ (09/09/2026)

**Bậc 1–2.** t5p · `latent_bottleneck` · λ=0.05 · **AdamW** · seed 42.
Nguồn Phase 1 thay bằng các pool dựng sẵn ở `data/pool/`; đích không đổi.

## B.1 — Hai lưới, và vì sao phải có lưới thứ hai

Lưới `pur*` chỉ điều khiển **độ tinh khiết** = tỉ lệ dòng nguồn có CWE nằm trong
4 CWE của đích (89, 78, 79, 22). Đo lại thành phần thật của từng file thì lộ ra
nó **không** cô lập được biến nào cả:

| pool | n | tinh khiết | **tỉ lệ js** | nhãn 1 |
|---|---|---|---|---|
| pur100_n930 | 930 | 1.000 | **0.873** | 0.500 |
| pur75_n930 | 930 | 0.751 | **0.676** | 0.500 |
| pur50_n930 | 930 | 0.500 | **0.481** | 0.500 |
| pur25_n930 | 930 | 0.249 | **0.316** | 0.500 |
| pur12_n930 | 930 | 0.120 | **0.192** | 0.500 |

Pha loãng độ tinh khiết **kéo tỉ lệ js sập theo** 0.873 → 0.192, vì các CWE ngoài
4 CWE đích hầu hết nằm ở phía ccpp. Vậy mọi Δ của lưới `pur*` là **hai biến cùng
đổi**, không đọc thành "hiệu ứng độ tinh khiết" được.

Lưới `lm*` (`src/build_langmatched_grid.py`) ghim tỉ lệ js ở 0.866–0.876 trong
khi độ tinh khiết vẫn rơi 1.00 → 0.12, n giữ 930, cân bằng nhãn giữ 0.50:

| pool | n | tinh khiet | tỉ lệ js | nhãn 1 |
|---|---|---|---|---|
| lm100_n930 | 930 | 1.000 | 0.873 | 0.500 |
| lm75_n930 | 930 | 0.751 | 0.875 | 0.499 |
| lm50_n930 | 930 | 0.500 | 0.866 | 0.501 |
| lm25_n930 | 930 | 0.249 | 0.872 | 0.501 |
| lm12_n930 | 930 | 0.120 | 0.876 | 0.500 |

`lm100_n930` và `pur100_n930` cùng thống kê nhưng **khác md5** (mẫu dòng khác),
nên mỗi lưới dùng neo của chính nó. Không bắc cầu.

## B.2 — Độ tinh khiết MỘT MÌNH: âm trên cả bốn chỉ số, và là BẬC THANG

Đối chứng `lm100_n930`, ghép cặp từng fold, **n=5 (đủ fold)**. `lm100_n930` khác
md5 với `data/phase1_4cwe.jsonl` nên đây là mẫu riêng, không phải nguồn `4cwe`.

| nhánh | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| lm75 | −0.0306 (0/5) | −0.0386 (0/5) | −0.0213 (0/5) | −0.0234 (0/5) |
| lm50 | −0.0118 (2/5) | −0.0191 (1/5) | −0.0175 (1/5) | −0.0230 (2/5) |
| lm25 | −0.0294 (0/5) | −0.0293 (0/5) | −0.0262 (1/5) | −0.0179 (1/5) |
| lm12 | −0.0358 (0/5) | −0.0480 (0/5) | −0.0310 (0/5) | −0.0298 (0/5) |

**16/16 ô đều âm.** Hai đầu (lm75 và lm12) chạm **sàn kiểm dấu ở n=5 — 0/5 fold
trên cả bốn chỉ số**. Biên độ −0.018…−0.031 ROC-AUC, trên sàn nhiễu 0.010.

### SỬA phát biểu ở n=4: không đơn điệu, mà là bậc thang

Ở n=4 tôi ghi hiệu ứng "đơn điệu theo mức pha loãng". Ở n=5 thì **sai**: ROC-AUC
đi −0.021 (75%) → −0.018 (50%) → −0.026 (25%) → −0.031 (12%), tức lm50 lại nhẹ
hơn lm75. So thẳng các mức với nhau thì gần như không có khác biệt nào:

| phép so | ΔROC-AUC | ΔF1@0.5 |
|---|---|---|
| lm12 − lm75 | −0.0098 (2/5) | −0.0051 (2/5) |
| lm25 − lm75 | −0.0049 (3/5) | +0.0012 (2/5) |
| lm50 − lm75 | +0.0038 (4/5) | +0.0188 (4/5) |
| lm12 − lm50 | −0.0136 (1/5) | −0.0240 (0/5) |

Chỉ cặp cực đoan nhất (12% so với 50%) mới nhích, và chỉ ở F1. **Phát biểu đúng:
rơi khỏi 100% là mất ~0.02–0.03 ROC-AUC; rơi bao nhiêu thì gần như không thêm.**
Độ tinh khiết nguồn hành xử gần như **được ăn cả ngã về không**.

## B.2b — Dòng lệch CWE CÓ HẠI, không phải độn vô hại

Đây là phép so sắc nhất của lưới, và nó chỉ làm được vì có sẵn trục kích thước.

- `lm75_n930` = **697 dòng đúng CWE + 233 dòng lệch CWE**
- `pur100_n465` = **465 dòng đúng CWE, không có dòng lệch nào**
- `pur100_n232` = **232 dòng đúng CWE**

Nếu dòng lệch CWE chỉ là độn vô hại thì `lm75` phải **thắng** — nó có nhiều hơn
50% dòng hữu ích. Ghép cặp từng fold, cùng máy, n=5:

| phép so | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| lm75 − pur100_n465 | −0.0174 (1/5) | −0.0123 (2/5) | −0.0161 (1/5) |
| lm75 − pur100_n232 | −0.0174 (1/5) | −0.0091 (1/5) | −0.0079 (1/5) |
| lm50 − pur100_n465 | +0.0014 (3/5) | −0.0085 (1/5) | −0.0158 (2/5) |
| lm12 − pur100_n232 | −0.0225 (1/5) | −0.0189 (2/5) | −0.0143 (1/5) |

**15/16 ô âm.** 697 dòng đúng CWE kèm 233 dòng lệch **thua 232 dòng đúng CWE một
mình** — một nguồn ít hơn ba lần về dòng hữu ích. Vậy thêm dữ liệu lỗ hổng lệch
lớp không phải "thêm dữ liệu", mà là **gây nhiễu chủ động**.

Đây là phát biểu mạnh nhất về transfer mà lưới này tạo ra: tri thức được chuyển
giao **mang tính CWE cụ thể**, và lớp CWE không khớp thì *can thiệp* chứ không
trung tính.

**BẬC 1–2, chưa kết luận.** Mọi ô ở bảng B.2b có p=0.375 (1–2/5 fold) — hướng thì
nhất quán nhưng số fold cùng dấu chưa đủ. Một máy duy nhất, và lưới `pur*` đã cho
thấy một máy có thể lệch tới 0.064. Bản lặp `pool_lm.sh|-|1 2 3` đã xếp trên
**ntat** và **158**; phải có nó rồi mới được viết.

## B.2c — Điểm val của Phase 1 KHÔNG dự báo được transfer (Spearman −0.191)

Phản biện hiển nhiên cho B.2b: "dòng lệch CWE làm **Phase 1 hỏng**, mất mát chỉ là
hệ quả của một checkpoint tệ." Bác được bằng dữ liệu đã có — trường
`phase1_val_macro_f1` nằm sẵn trong mọi ô, không cần chạy lại gì.

11 pool trên `ntat2`, ΔROC-AUC ghép cặp từng fold so với `pur100_n930`:

| pool | Phase 1 val | ΔROC-AUC |
|---|---|---|
| lm25_n930 | **0.6867** ← cao nhất | **−0.0221** |
| lm100_n930 | 0.6458 | **+0.0041** ← tốt nhất |
| lm50_n930 | 0.6417 | −0.0133 |
| pur50_n930 | 0.6246 | −0.0301 |
| pur75_n930 | 0.6190 | +0.0118 |
| lm75_n930 | 0.6186 | −0.0171 |
| lm12_n930 | 0.6046 | −0.0269 |
| pur100_n465 | 0.5943 | −0.0049 |
| pur25_n930 | 0.5636 | −0.0007 |
| pur12_n930 | 0.5500 | −0.0082 |
| pur100_n232 | **0.4933** ← thấp nhất | −0.0080 |

**Spearman(Phase 1 val, ΔROC) = −0.191** — bằng không, hơi âm.

Cặp đối lập rõ nhất: `lm25` học nguồn của chính nó **tốt nhất trong cả 11 pool**
(0.6867) mà transfer **−0.0221**; `pur100_n232` học tệ nhất (0.4933, kém 0.19)
mà transfer chỉ **−0.0080** — tốt gần gấp ba.

### Hai điều rút ra

1. **B.2b không phải hệ quả của Phase 1 hỏng.** `lm75` (Phase 1 0.6186) thua
   `pur100_n232` (Phase 1 0.4933) ở transfer, dù Phase 1 hơn 0.125. Cơ chế nằm ở
   **thành phần nguồn**, không ở chất lượng khớp nguồn.
2. **Không được chọn nguồn transfer bằng điểm Phase 1** — đó là cách chọn tự
   nhiên nhất và nó sai. Phải chọn bằng **độ trùng không gian nhãn** với đích.

### Giới hạn phải nêu kèm

Mỗi pool có tập val **của riêng nó**, nên điểm val giữa các pool không so trực
tiếp được: 0.6867 của `lm25` là trên một bài dễ hơn. Nhưng đó **chính là lý do**
con số ấy vô dụng cho việc chọn nguồn — người chọn nguồn chỉ có đúng con số đó
cho từng ứng viên, và nó dẫn họ đi sai. Vẫn là n=11 pool, một máy, một backbone.

## B.2d — Tách theo nhóm rò rỉ: B.2 MẠNH LÊN, B.2b PHẢI RÚT (09/09)

Cùng phép kiểm đã dùng cho trục ρ (FACTS §21.1): bộ `sven_python_folds_norm` chia
theo từng dòng nên ~16% hàng test có bản đối nghịch gần trùng trong TRAIN. Các ô
pool đều có `test_probabilities` nên tách được **không chạy lại gì**. Hai máy
(ntat2 + 161), ghép cặp từng fold trong-máy.

### Độ tinh khiết: hiệu ứng SỐNG trên hàng sạch, và là LIỀU–ĐÁP ỨNG chứ không bậc thang

| phép so | nhóm rò rỉ (~16%) ΔROC | **nhóm SẠCH (~73%) ΔROC** | nhóm sạch ΔF1 |
|---|---|---|---|
| lm75 − lm100 | −0.0504 (2/8) | −0.0010 (3/8) | −0.0168 (1/8) |
| lm50 − lm100 | −0.0763 (1/7) | −0.0093 (2/7) | −0.0116 (1/7) |
| lm25 − lm100 | −0.0894 (0/7) | **−0.0197 (0/7, p=0.016)** | **−0.0296 (0/7, p=0.016)** |
| lm12 − lm100 | −0.1174 (0/7) | **−0.0278 (0/7, p=0.016)** | **−0.0436 (0/7, p=0.016)** |

Pha loãng **nặng** (25%, 12%) làm hỏng khả năng khái quát hoá thật: trên 73% hàng
không có bản gần trùng nào, **0/7 fold dương trên cả hai chỉ số**, biên độ −0.020…
−0.028 ROC-AUC — trên sàn nhiễu 0.010.

**SỬA phát biểu "bậc thang" ở B.2.** Trên hàng sạch, ΔROC đi −0.001 → −0.009 →
−0.020 → −0.028 theo mức pha loãng: **đơn điệu, liều–đáp ứng rõ ràng**. Dáng bậc
thang trong số tổng là **hiện vật của nhóm rò rỉ** — nhóm đó chịu −0.05 ngay từ
lm75 rồi bão hoà. Tách nhóm ra thì hình dạng thật lộ ra, và nó đẹp hơn.

### B.2b KHÔNG sống trên hàng sạch — rút lại

| phép so | nhóm rò rỉ ΔROC | **nhóm SẠCH ΔROC** | nhóm sạch ΔF1 |
|---|---|---|---|
| lm75 − pur100_n232 | −0.0088 (2/8) | **−0.0013 (5/8)** | **+0.0038 (4/8)** |

Phát biểu "697 dòng đúng CWE kèm 233 dòng lệch **thua** 232 dòng đúng CWE một mình"
(số tổng −0.0174 F1 / −0.0091 ROC) **biến mất hoàn toàn trên 73% hàng sạch**: ROC
−0.0013 (5/8), F1 **+0.0038** (4/8). Nó là hiệu ứng của nhóm rò rỉ.

**Vậy không được nói "dòng lệch CWE gây nhiễu chủ động".** Phát biểu còn đứng được là
yếu hơn nhưng vẫn có giá trị:

> Ở mức pha loãng nặng (≤25% dòng đúng CWE), nguồn transfer mất khả năng giúp mô
> hình khái quát hoá sang mã chưa từng thấy — đo trên 73% hàng test không có bản
> gần trùng, 0/7 fold, cả F1 lẫn ROC-AUC. Ở mức pha loãng nhẹ (75%, 50%) thì
> **không phân biệt được với nhiễu** trên hàng sạch.

### Vì sao mục này quan trọng hơn bản thân con số

Đây là **lần thứ hai trong một đêm** phép tách nhóm rò rỉ đổi một phát biểu tiêu đề
(lần đầu: FACTS §21.1 với trục ρ). Cả hai lần, số tổng đều bị nhóm ~16% hàng dễ học
vẹt kéo. Với bộ `sven_python_folds_norm` chia theo dòng, **mọi Δ tổng đều phải kèm
phép tách này trước khi được gọi là phát hiện.**

**Giới hạn:** n=7–8 (một số fold không đủ 8 hàng trong nhóm nên bị bỏ). Ở n=7 thì
p=0.016 là **sàn** — nghĩa là "cùng dấu ở cả 7", không phải "rất có ý nghĩa".

## B.2e — Bản lặp trên máy thứ hai ĐÃ ĐỦ n=5: chỉ pha loãng NẶNG lặp lại

Cả hai máy đều n=5, mỗi máy dùng đối chứng `lm100_n930` của chính nó.

| nhánh | ntat2 ΔROC | 161 ΔROC | ntat2 ΔF1@0.5 | 161 ΔF1@0.5 | lặp? |
|---|---|---|---|---|---|
| lm75 | −0.0213 (0/5) | **+0.0028** (2/5) | −0.0306 (0/5) | −0.0023 (1/5) | **KHÔNG** |
| lm50 | −0.0175 (1/5) | **+0.0054** (2/5) | −0.0118 (2/5) | −0.0001 (3/5) | **KHÔNG** |
| lm25 | −0.0262 (1/5) | −0.0188 (1/5) | −0.0294 (0/5) | −0.0276 (1/5) | CÓ |
| **lm12** | −0.0310 (**0/5**) | −0.0400 (**0/5**) | −0.0358 (**0/5**) | −0.0420 (**0/5**) | **CÓ, cả hai chạm sàn** |

Gộp **dấu** của hai máy (mỗi Δ vẫn ghép cặp trong-máy, nên đây là gộp hai bản lặp
độc lập chứ không phải trộn phần cứng):

| nhánh | F1@0.5 | ROC-AUC |
|---|---|---|
| **lm12** | **0/10, p=0.0020** | **0/10, p=0.0020** |
| lm25 | 1/10, p=0.0215 | 2/10, p=0.1094 |
| lm50 | 5/10, p=1.000 | 3/10, p=0.3438 |
| lm75 | 1/10, p=0.0215 | 2/10, p=0.1094 |

`lm12` là ô duy nhất **âm ở cả 10/10 fold trên cả F1 lẫn ROC-AUC**, hai máy độc lập.
`lm75` âm ở F1 (1/10) nhưng **không** ở ROC (2/10, và 161 đổi dấu) — đúng kiểu hiệu
ứng chỉ ở ngưỡng chứ không ở xếp hạng, và B.2d đã cho thấy nó null trên hàng sạch.

**Khớp với B.2d, không mâu thuẫn.** B.2d đo trên 73% hàng không rò rỉ: lm75 −0.0010
và lm50 −0.0093 ROC (3/8 và 2/7) — null. Hiệu ứng null thì dấu phụ thuộc rút thăm
fold và máy, đúng như quan sát ở đây.

**Phải rút** phần "16/16 ô âm, `lm75` 0/5 trên cả bốn chỉ số" ở B.2: đúng trên ntat2
nhưng không lặp lại. Ở n=5 một máy, "0/5" là **sàn** kiểm dấu — nó không phân biệt
được hiệu ứng thật với may mắn (CLAUDE.md §2b).

### Phát biểu cuối, sau ba phép kiểm độc lập

> Pha loãng nguồn Phase 1 xuống **≤25% dòng có CWE trùng đích** làm hỏng khả năng
> khái quát hoá của mô hình đích. Ở mức 12%: **0/10 fold** trên cả F1@0.5 lẫn
> ROC-AUC qua **hai máy độc lập** (p=0.002 mỗi chỉ số), biên độ −0.031…−0.040
> ROC-AUC — gấp 3–4 lần sàn nhiễu. Hiệu ứng **sống trên 73% hàng test không có bản
> gần trùng** (0/7 fold, cả hai chỉ số). Pha loãng **nhẹ (75%, 50%) không phân biệt
> được với nhiễu**: null trên hàng sạch và đổi dấu giữa hai máy.
>
> Tỉ lệ ngôn ngữ được **ghim ở js 0.87** trong toàn lưới, nên đây là hiệu ứng của
> **trùng lớp CWE**, không phải của khoảng cách ngôn ngữ.

## B.3 — Lưới `pur*` KHÔNG đơn điệu, và một nửa không lặp lại được

Hai máy độc lập, mỗi máy dùng đối chứng `pur100_n930` của chính nó (ROC-AUC):

| nhánh | 161 | ntat2 | lặp lại? |
|---|---|---|---|
| pur75 | −0.0021 (2/5) | +0.0118 (4/5) | ~0 ở cả hai — **không có hiệu ứng** |
| pur50 | **−0.0450 (0/5)** | **−0.0301 (0/5)** | **CÓ** — 0/5 ở cả hai máy |
| pur25 | −0.0646 (0/5) | −0.0007 (1/5) | **KHÔNG** — lệch 0.064, gấp 2.3× sàn liên-GPU |
| pur12 | +0.0026 (2/5) | −0.0082 (2/5) | ~0 ở cả hai |
| pur100_n465 | −0.0122 (1/5) | −0.0049 (2/5) | CÓ, nhỏ |
| pur100_n232 | −0.0226 (1/5) | −0.0080 (2/5) | CÓ, nhỏ |

Đường cong `pur*` gấp khúc chứ không đơn điệu, và điểm gấp mạnh nhất (pur25)
**không lặp lại được**. So với B.2 thì rõ nguyên nhân: ở pur25/pur12 tỉ lệ js đã
tụt còn 0.32/0.19, nên hai biến kéo ngược nhau và cái nào thắng phụ thuộc fold.

**Rút lại một phát biểu tạm của tối 08/09.** Tôi đã ghi "khi ghim tỉ lệ js thì
hiệu ứng độ tinh khiết gần như biến mất, có thể biến thật là ngôn ngữ" — đó là
đọc ở n=2–3. Ở n=4 thì **ngược lại**: ghim ngôn ngữ làm hiệu ứng độ tinh khiết
*sạch hơn*, còn ngôn ngữ chính là cái nhiễu đã làm gấp khúc lưới `pur*`.

## B.4 — Kích thước nguồn: có tác dụng, độc lập với độ tinh khiết

Giữ độ tinh khiết 100%, cắt n 930 → 465 → 232 (tỉ lệ js gần như không đổi
0.873/0.858/0.853): ROC-AUC −0.0049…−0.0122 (n465) và −0.0080…−0.0226 (n232),
**âm trên cả hai máy**. Nhỏ nhưng nhất quán. Vậy Δ của các nhánh pha loãng không
quy về "ít dữ liệu đích hơn" được — n giữ nguyên 930 ở toàn lưới.

## B.5 — Ngôn ngữ của phần pha loãng, ở cùng độ tinh khiết (SÀNG LỌC, n=4)

So thẳng `lm_X` với `pur_X`: cùng n, cùng độ tinh khiết, chỉ khác **các dòng pha
loãng là js hay ccpp**. Δ âm nghĩa là pha loãng bằng **ccpp tốt hơn** js.

| cặp | ΔROC-AUC | ΔPR-AUC |
|---|---|---|
| lm75 − pur75 | −0.0285 (0/4) | −0.0168 (2/4) |
| lm50 − pur50 | +0.0115 (3/4) | +0.0075 (2/4) |
| lm25 − pur25 | −0.0237 (1/4) | −0.0158 (1/4) |
| lm12 − pur12 | −0.0200 (1/4) | −0.0115 (1/4) |

Ba trên bốn mức nói **ccpp là phần pha loãng tốt hơn js**, dù đích là Python và
js gần Python hơn về cú pháp bề mặt. Điều này khớp với việc nguồn `full` (chủ yếu
ccpp) vẫn chạy được. **Đây là bậc 1, p ≥ 0.125 ở mọi ô — chưa kết luận gì.**

## B.6 — Còn thiếu gì trước khi viết được

1. `lm*` fold 5 — đang chạy trên ntat2.
2. `lm*` trên **máy thứ hai** (`pool_lm.sh|-|1 2 3` đã xếp trên ntat). Lưới `pur*`
   cho thấy vì sao bắt buộc: pur25 lệch 0.064 giữa hai máy.
3. `poolcb_codebert` để biết hiệu ứng có phụ thuộc backbone không (12/60 ô).
