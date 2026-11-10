"""The first N_FILES of the 14 FineWeb-Edu sample-10BT parquet files, pretokenized by the unmodified fineweb.py
(Llama 2 tokenizer, ~100M-token uint16 shards, last shard = validation) into DATA_DIR. Training imports this module
in place of fineweb so both sides agree on DATA_DIR. Run from the repository root:

$ python scripts/fineweb/fineweb_subset.py pretokenize
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import fineweb  # noqa: E402

N_FILES = 2
fineweb.PATTERN = "sample/10BT/00[0-1]_*.parquet"
fineweb.DATA_DIR = os.path.join(fineweb.DATA_CACHE_DIR, f"FineWebEdu_{N_FILES}of14")
Task = fineweb.Task

if __name__ == "__main__" and sys.argv[1:] == ["pretokenize"]:
    files = fineweb.parquet_files()
    print("parquet files:", [os.path.basename(f) for f in files])
    assert len(files) == N_FILES, files
    fineweb.pretokenize(num_workers=N_FILES)
