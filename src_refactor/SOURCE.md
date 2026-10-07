# src_refactor — mã nhánh `refactor` của maytinhdibo/GraphTransferVD (thư mục refactorMWG/)

- commit: `c1f16417c52949a7967dabaadff6ae8b165c2034` (2026-09-27 01:25:48 +0700 refactorMWG: chọn checkpoint theo train_loss; --also_select giữ thêm checkpoint thứ hai)
- lấy ngày 2026-09-28 01:02:25 bằng `git archive c1f16417c52949a7967dabaadff6ae8b165c2034 refactorMWG`, BỎ `data/` (SVEN kèm theo — khối final dùng data/final_experiment_data/sven_python_folds_nocomment).
- dùng cho tab "refactor" của _FinalPaperExperiment: Pha 1 chọn checkpoint theo train loss (--selection_metric train_loss) + thử lại khi kẹt (--stuck_epoch 3, --stuck_min_drop 0.02, scripts/p1_retry.py).
- KHÔNG sửa file nào trong thư mục này; mọi khác biệt với src_final là của nhánh refactor.

## md5 (LC_ALL=C sort)

```
611bbc2e3877801995b45a2bcfe0abd4  ./.gitignore
3a678a872fc57831509fd300a1020bef  ./README.md
c291e54eeede61c31afe439c6d04ead9  ./dataset/EXPECTED.md5
8a6fd773963b6dfbf4d6d7aa00badb73  ./dataset/build_pools.py
0ac83a276d395dea222160534149e525  ./dataset/build_sources.py
d8534d8fdb5d87e6cd1be836d68925b3  ./dataset/cwe.py
9cfc07051dc696769f70e8fe4bec1739  ./dataset/cwe_root_parents.txt
fb39bbfb2e7b2e20a2556e836861e1f4  ./dataset/export_clean_sources.py
1b5a3e67fe1ff2f0435807d9091a4fd9  ./dataset/filters.py
f1d3f751308b5c2b4bff61766552a1a7  ./dataset/make_target.py
a1421eddd16d449c2d781f5039bade10  ./mwg/__init__.py
fa4c1340bd09d096960b1f42d4f1fa56  ./mwg/comments.py
b2cf0dd29bda06232999274691ceb7b7  ./mwg/data.py
0356bd242af02b96e1995a0aea6f1460  ./mwg/graph.py
2495d7c755fd3cf7a1a45f14e359fc4b  ./mwg/metrics.py
b6fe234ba2433de5b2596e4ce97f660c  ./mwg/model.py
bcc34ca1c345089ede9ec48af48754ab  ./mwg/optim.py
065b5ace0f9914ad6f91e0a73d0dfd9b  ./mwg/symbols.py
9946a0bbbfcf245e36a8ae786f4cb8dc  ./mwg/utils.py
181819fad1bc934015918871cc8f5013  ./requirements.txt
e4ee4412225bda04ffd299c72153e020  ./scripts/build_data.sh
8f8aa74ed08b94a0e573a88e3098b50e  ./scripts/run_p1.sh
96873272b7d6a5a3d0f7d8f2c6651331  ./scripts/run_sota.sh
af301e28556ab33c5d200f53ab424267  ./scripts/run_target_only.sh
575eebce8f2f31b3144f49970d137b2d  ./scripts/summarize.py
ea518a22cf5a78f4138820df95439b14  ./tests/test_stuck.py
2c06c340d89c9c650a7c250f6418f82a  ./tests/test_symbols.py
ddb9e37883c5e8fb025f096cf878b2f9  ./train.py
```
