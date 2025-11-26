"""FlowTCR-Gen inference pipeline entrypoint."""

from flowtcr_fold.inference.pipeline import run_pipeline

if __name__ == "__main__":
    cands, scores = run_pipeline()
    print("Generated", cands.shape[0], "candidates")
