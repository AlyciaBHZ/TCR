
import torch
from flowtcr_fold.Immuno_PLM.immuno_plm import ImmunoPLM

def test_shape():
    B = 2
    L = 393
    z_dim = 128
    hidden_dim = 256
    
    model = ImmunoPLM(hidden_dim=hidden_dim, z_dim=z_dim, use_esm=False)
    
    tokens = torch.randint(0, 100, (B, L))
    mask = torch.ones(B, L)
    
    # Mock region slices
    # Sample 0 has regions, Sample 1 has regions
    region_slices = [
        {"cdr3b": slice(10, 20), "mhc": slice(30, 40)},
        {"cdr3b": slice(15, 25), "mhc": slice(35, 45)}
    ]
    
    print("Running forward pass...")
    try:
        out = model(tokens, mask, region_slices=region_slices)
        print("Success!")
        print("Pooled shape:", out["pooled"].shape)
    except Exception as e:
        print("Failed!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shape()
