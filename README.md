Backup MultiVD from 161 (2026-10-29), without model/ checkpoints and secret files.

Restore:
  cat part1/multivd_code_results.tar.zst.* | zstd -d | tar -xf -
  cat part2/multivd_data.tar.zst.* | zstd -d | tar -xf -
