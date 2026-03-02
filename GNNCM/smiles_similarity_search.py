import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch, Data

from models.gcl_model import GCLModel

LIBRARY_SMILES_PATH = r"data\1_million.txt"
ANCHOR_SMILES_PATH = r"data\anchor_smiles.txt"
MODEL_PATH = r"results\checkpoints\gcn_augmentations_0_2_best_model.pth"
OUTPUT_TXT_PATH = r"results/similarity/Filtered_results_0.95_1_million.txt"
DEFAULT_SIM_THRESHOLD = 0.95

MODEL_CONFIG = {
    "encoder_name": "gcn",
    "input_dim": 19,
    "hidden_dim": 256,
    "projection_dim": 128,
    "num_layers": 5,
    "dropout": 0.5,
    "temperature": 0.1,
}

ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H"]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


@dataclass
class MoleculeRecord:
    mol_id: str
    smiles: str
    mol: Chem.Mol


def auto_prefix(path: str, fallback: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base if base else fallback


def read_smiles_file(
    path: str,
    prefix: str,
    progress_label: Optional[str] = None,
    progress_interval: int = 1000,
) -> List[MoleculeRecord]:
    records: List[MoleculeRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            lowered = raw.lower()
            if lowered in {"smiles", "pid", "pid smiles", "pid,smiles", "pid\tsmiles"}:
                continue
            pid: Optional[str] = None
            smiles = raw
            if "\t" in raw:
                pid_part, smiles_part = raw.split("\t", 1)
                pid = pid_part.strip() or None
                smiles = smiles_part.strip()
            elif "," in raw:
                pid_part, smiles_part = raw.split(",", 1)
                pid = pid_part.strip() or None
                smiles = smiles_part.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: failed to parse SMILES on line {idx + 1}; skipped.")
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                print(f"Warning: molecule sanitization failed on line {idx + 1}; skipped.")
                continue
            mol_id = pid if pid else f"{prefix}_{idx:05d}"
            records.append(MoleculeRecord(mol_id=mol_id, smiles=Chem.MolToSmiles(mol), mol=mol))
            if progress_label and (idx + 1) % progress_interval == 0:
                print(f"{progress_label}: processed {idx + 1} lines")
    if progress_label:
        print(f"{progress_label}: finished reading {len(records)} valid molecules")
    return records


def atom_feature_vector(atom: Chem.Atom) -> List[float]:
    features: List[float] = []
    symbol = atom.GetSymbol()
    features.extend(1.0 if symbol == t else 0.0 for t in ATOM_TYPES)
    features.append(float(atom.GetDegree()))
    features.append(float(atom.GetFormalCharge()))
    features.extend(1.0 if atom.GetHybridization() == h else 0.0 for h in HYBRIDIZATION_TYPES)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(float(atom.GetTotalNumHs()))
    return features


def bond_feature_vector(bond: Chem.Bond) -> List[float]:
    features: List[float] = []
    features.extend(1.0 if bond.GetBondType() == t else 0.0 for t in BOND_TYPES)
    features.append(1.0 if bond.IsInRing() else 0.0)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    return features


def mol_to_graph(record: MoleculeRecord) -> Optional[Data]:
    mol = record.mol
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_features = [atom_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_feature_vector(bond)
        edge_indices.extend([(begin, end), (end, begin)])
        edge_attrs.extend([feat, feat])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.mol_id = record.mol_id
    data.smiles = record.smiles
    return data


def molecules_to_graphs(
    records: Sequence[MoleculeRecord],
    progress_label: Optional[str] = None,
    progress_interval: int = 10000,
) -> List[Data]:
    graphs: List[Data] = []
    for record in records:
        graph = mol_to_graph(record)
        if graph is not None:
            graphs.append(graph)
        if progress_label and len(graphs) % progress_interval == 0:
            print(f"{progress_label}: built {len(graphs)} graphs")
    if progress_label:
        print(f"{progress_label}: completed with {len(graphs)} graphs")
    return graphs


def build_model(model_path: str, device: torch.device) -> GCLModel:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = GCLModel(
        encoder_name=MODEL_CONFIG["encoder_name"],
        input_dim=MODEL_CONFIG["input_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        projection_dim=MODEL_CONFIG["projection_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        dropout=MODEL_CONFIG["dropout"],
        temperature=MODEL_CONFIG["temperature"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_embeddings(
    model: GCLModel,
    graphs: Sequence[Data],
    device: torch.device,
    batch_size: int,
    progress_label: Optional[str] = None,
) -> torch.Tensor:
    if not graphs:
        hidden_dim = MODEL_CONFIG["hidden_dim"]
        return torch.empty((0, hidden_dim))

    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        total = len(graphs)
        for start in range(0, total, batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)
            embedding = model.get_embedding(batch).detach().cpu()
            embeddings.append(embedding)
            if progress_label:
                processed = min(start + batch_size, total)
                print(f"{progress_label}: completed {processed}/{total}")
    return torch.cat(embeddings, dim=0)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_similarity_matrix(library_embeddings: torch.Tensor, anchor_embeddings: torch.Tensor) -> torch.Tensor:
    if library_embeddings.numel() == 0 or anchor_embeddings.numel() == 0:
        return torch.zeros((library_embeddings.size(0), anchor_embeddings.size(0)))
    library_norm = F.normalize(library_embeddings, p=2, dim=1)
    anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
    return library_norm @ anchor_norm.t()


def collect_threshold_matches(
    sims: torch.Tensor,
    library_records: Sequence[MoleculeRecord],
    threshold: float,
    exclude_self: bool,
    anchor_record: MoleculeRecord,
) -> List[Tuple[MoleculeRecord, float]]:
    matches: List[Tuple[MoleculeRecord, float]] = []
    if sims.numel() == 0:
        return matches

    indices = torch.nonzero(sims >= threshold, as_tuple=False).view(-1).tolist()
    for idx in indices:
        sim = sims[idx].item()
        if sim < threshold:
            continue
        record = library_records[idx]
        if exclude_self and record.mol_id == anchor_record.mol_id:
            continue
        matches.append((record, sim))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def write_similarity_report(
    output_txt: str,
    anchor_records: Sequence[MoleculeRecord],
    library_records: Sequence[MoleculeRecord],
    sim_matrix: torch.Tensor,
    threshold: float,
    exclude_self: bool,
) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)
    written = 0
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write(f"Similarity threshold: {threshold:.2f}\n")
        handle.write("=" * 60 + "\n\n")
        for anchor_idx, anchor in enumerate(anchor_records):
            sims = sim_matrix[:, anchor_idx]
            matches = collect_threshold_matches(sims, library_records, threshold, exclude_self, anchor)
            if not matches:
                continue
            handle.write(f"Anchor molecule: {anchor.mol_id} | {anchor.smiles}\n")
            for rank, (record, score) in enumerate(matches, start=1):
                handle.write(
                    f"  {rank:02d}. Library molecule: {record.mol_id} | {record.smiles} | Similarity: {score:.4f}\n"
                )
                written += 1
            handle.write("\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter library molecules whose similarity to anchor molecules exceeds the threshold."
    )
    parser.add_argument(
        "--anchor_smiles",
        default=ANCHOR_SMILES_PATH,
        help="Input SMILES text file for anchor molecules.",
    )
    parser.add_argument(
        "--library_smiles",
        default=LIBRARY_SMILES_PATH,
        help="Input SMILES text file for the search library.",
    )
    parser.add_argument(
        "--model_path",
        default=MODEL_PATH,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_txt",
        default=OUTPUT_TXT_PATH,
        help="Output path for the filtered TXT report.",
    )
    parser.add_argument("--device", default="auto", help="Inference device. Use auto to prefer CUDA.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size used during embedding inference.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=DEFAULT_SIM_THRESHOLD,
        help="Minimum similarity threshold for retaining a match.",
    )
    parser.add_argument(
        "--exclude_self",
        action="store_true",
        help="Exclude self-matches when anchor and library inputs are the same.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.anchor_smiles):
        raise FileNotFoundError(f"Anchor file not found: {args.anchor_smiles}")

    library_path = args.library_smiles
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"Library file not found: {library_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading model checkpoint...")
    model = build_model(args.model_path, device)

    anchor_prefix = auto_prefix(args.anchor_smiles, "anchor") + "_a"
    lib_prefix = auto_prefix(library_path, "library") + "_lib"

    print("Reading anchor molecules from SMILES text...")
    anchor_records = read_smiles_file(
        args.anchor_smiles,
        anchor_prefix,
        progress_label="Anchor read progress",
        progress_interval=500,
    )
    print(f"Anchor molecule count: {len(anchor_records)}")

    print("Reading library molecules from SMILES text...")
    library_records = read_smiles_file(
        library_path,
        lib_prefix,
        progress_label="Library read progress",
        progress_interval=2000,
    )
    print(f"Library molecule count: {len(library_records)}")

    if not anchor_records:
        raise RuntimeError("No valid anchor molecules were parsed from the input file.")
    if not library_records:
        raise RuntimeError("No valid library molecules were parsed from the input file.")

    reuse_library = os.path.abspath(args.anchor_smiles) == os.path.abspath(library_path)

    print("Building graph data...")
    anchor_graphs = molecules_to_graphs(
        anchor_records,
        progress_label="Anchor graph build progress",
        progress_interval=100,
    )
    print(f"Anchor graph count: {len(anchor_graphs)}")
    if reuse_library:
        library_graphs = anchor_graphs
        print("Reusing anchor graphs for the library input.")
    else:
        library_graphs = molecules_to_graphs(
            library_records,
            progress_label="Library graph build progress",
            progress_interval=1000,
        )
        print(f"Library graph count: {len(library_graphs)}")

    if not anchor_graphs:
        raise RuntimeError("Failed to build anchor graphs.")
    if not library_graphs:
        raise RuntimeError("Failed to build library graphs.")

    print("Computing anchor embeddings...")
    anchor_embeddings = compute_embeddings(
        model,
        anchor_graphs,
        device,
        args.batch_size,
        progress_label="Anchor embedding progress",
    )

    print("Computing library embeddings...")
    if reuse_library:
        library_embeddings = anchor_embeddings
        print("Reusing anchor embeddings for the library input.")
    else:
        library_embeddings = compute_embeddings(
            model,
            library_graphs,
            device,
            args.batch_size,
            progress_label="Library embedding progress",
        )
        print("Library embedding computation finished.")

    print("Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(library_embeddings, anchor_embeddings)
    print("Similarity matrix computed.")

    if args.similarity_threshold <= 0:
        print("Warning: invalid similarity threshold supplied; resetting to 0.95.")
        args.similarity_threshold = DEFAULT_SIM_THRESHOLD

    print(f"Writing results to {args.output_txt}...")
    rows = write_similarity_report(
        output_txt=args.output_txt,
        anchor_records=anchor_records,
        library_records=library_records,
        sim_matrix=sim_matrix,
        threshold=args.similarity_threshold,
        exclude_self=args.exclude_self,
    )
    print("Result report written.")
    print(f"Matching rows retained: {rows}")
    print("Similarity search finished.")


if __name__ == "__main__":
    main()
