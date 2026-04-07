## Benchmark Results

The benchmark result files (`.sqlite`) are hosted on Hugging Face Datasets due to their size:

🤗 [ShinkaEvolve-benchmark Dataset](https://huggingface.co/datasets/diegobragaferreira/ShinkaEvolve-benchmark)

### Restoring the benchmark data

#### Option 1 — Python (recommended)
Install the required library:
```bash
pip install huggingface_hub
```

Then run
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="diegobragaferreira/ShinkaEvolve-benchmark",
    repo_type="dataset",
    local_dir="./results"
)
```

Option 2 - command line
```bash
pip install huggingface_hub

huggingface-cli download diegobragaferreira/ShinkaEvolve-benchmark \
  --repo-type dataset \
  --local-dir ./results
```

After downloading, the .sqlite files will be restored inside the results/ folder,
matching the expected structure of this repository.
