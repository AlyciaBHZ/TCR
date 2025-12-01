"""
Fetch protein sequences for unmapped gene IDs via NCBI E-utilities.

Reads `detailed_unmapped_ids.csv` (columns: column, original_id, mapped_to),
searches each unique `original_id` as a gene term in the NCBI protein
database, and writes a summary CSV `unmapped_protein_sequences.csv` with:
`original_id`, `protein_accession`, `sequence`, `status`.

Notes:
- Requires internet access.
- Set environment variable `NCBI_EMAIL` (and optionally `NCBI_API_KEY`) to
  comply with NCBI usage policy and improve rate limits.
- The search term is `<gene_id>[gene]`; adjust if your IDs need a different
  query (e.g., organism filters).
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
INPUT_PATH = Path("data/collected data/final_data/detailed_unmapped_ids.csv")
OUTPUT_PATH = Path("data/collected data/final_data/unmapped_protein_sequences.csv")

# Be polite to NCBI; adjust if you have an API key (higher rate limits).
SLEEP_BETWEEN_CALLS = 0.34


def entrez_request(endpoint: str, params: Dict[str, str], retmode: str = "json") -> str:
    """Make an HTTP GET to an NCBI E-utilities endpoint and return raw text."""
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("email", os.environ.get("NCBI_EMAIL", "example@example.com"))
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    if retmode:
        params.setdefault("retmode", retmode)

    query = urllib.parse.urlencode(params)
    url = f"{ENTREZ_BASE}{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def search_protein_ids(gene_term: str, retmax: int = 3) -> List[str]:
    """Search protein IDs by gene name."""
    payload = {
        "db": "protein",
        "term": f"{gene_term}[gene]",
        "retmax": str(retmax),
    }
    raw = entrez_request("esearch.fcgi", payload, retmode="json")
    data = json.loads(raw)
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_protein_fasta(protein_id: str) -> Tuple[str, str]:
    """Fetch a protein FASTA by protein ID; return (accession, sequence)."""
    raw = entrez_request(
        "efetch.fcgi",
        {"db": "protein", "id": protein_id, "rettype": "fasta", "retmode": "text"},
        retmode="text",
    )
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("Invalid FASTA response")
    header = lines[0][1:]
    accession = header.split()[0]
    sequence = "".join(lines[1:]).strip()
    if not sequence:
        raise ValueError("Empty sequence in FASTA")
    return accession, sequence


def load_unmapped_ids(path: Path) -> List[str]:
    """Read unique gene IDs from detailed_unmapped_ids.csv."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df_rows: List[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = (row.get("original_id") or "").strip()
            if gene:
                df_rows.append(gene)
    return sorted(set(df_rows))


def fetch_sequences_for_genes(genes: Iterable[str]) -> List[Dict[str, str]]:
    """Fetch protein sequences for each gene name; returns result rows."""
    results: List[Dict[str, str]] = []
    for gene in genes:
        status = "not_found"
        accession = ""
        sequence = ""
        try:
            ids = search_protein_ids(gene)
            time.sleep(SLEEP_BETWEEN_CALLS)
            if ids:
                acc, seq = fetch_protein_fasta(ids[0])
                accession, sequence, status = acc, seq, "ok"
            else:
                status = "no_protein_ids"
        except Exception as exc:  # noqa: BLE001
            status = f"error: {exc}"
        results.append(
            {
                "original_id": gene,
                "protein_accession": accession,
                "sequence": sequence,
                "status": status,
            }
        )
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results


def write_results(rows: List[Dict[str, str]], path: Path) -> None:
    """Write summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["original_id", "protein_accession", "sequence", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    genes = load_unmapped_ids(INPUT_PATH)
    print(f"Loaded {len(genes)} unique gene IDs from {INPUT_PATH}")
    rows = fetch_sequences_for_genes(genes)
    write_results(rows, OUTPUT_PATH)
    print(f"Wrote summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
