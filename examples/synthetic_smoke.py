"""Run a labelled synthetic smoke check. No real medical data is used."""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.synthetic_data import make_synthetic_stroke
from stroke_tree import run_complete_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="synthetic_outputs")
    args = parser.parse_args()
    print("SYNTHETIC SOFTWARE CHECK ONLY. These are not historical or medical results.")
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "synthetic.csv"
        make_synthetic_stroke().to_csv(csv, index=False)
        result = run_complete_analysis(csv, args.output_dir, show_plots=False)
    manifest = Path(args.output_dir) / "run_manifest.json"
    import json
    data = json.loads(manifest.read_text())
    data["dataset_kind"] = "synthetic software-check fixture; not patient data"
    manifest.write_text(json.dumps(data, indent=2))
    print("Synthetic smoke check completed; do not present these metrics as project performance.")

