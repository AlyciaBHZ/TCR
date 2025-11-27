"""FlowTCR-Gen inference pipeline entrypoint."""

from flowtcr_fold.FlowTCR_Gen.pipeline_impl import run_pipeline


if __name__ == "__main__":
    cands, scores = run_pipeline()
    print("Generated", len(cands) if isinstance(cands, list) else cands.shape[0], "candidates")
