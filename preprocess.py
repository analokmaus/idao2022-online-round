import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from pymatgen.core import Structure
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, default=4, help='cutoff radius for graph converter')
    parser.add_argument('--input_path', type=str, default='data/')
    parser.add_argument('--output_path', type=str, default='data/')
    args = parser.parse_args()

    train_dir = Path(args.input_path) / 'dichalcogenides_public'
    test_dir = Path(args.input_path) / 'dichalcogenides_private'

    train_json_path = list((train_dir / 'structures').glob('*.json'))
    test_json_path = list((test_dir / 'structures').glob('*.json'))
    train = pd.read_csv(train_dir/'targets.csv')

    converter = CrystalGraph(
        bond_converter=GaussianDistance(centers=np.linspace(0, args.cutoff+1, 100), width=0.5),
        cutoff=args.cutoff
    )
    print(f'Cutoff is {args.cutoff}')

    cache = []
    for p in tqdm(train_json_path):
        record = {}
        target = train.loc[train['_id'] == p.stem, 'band_gap'].values
        structure = read_pymatgen_dict(p)
        result = converter(structure)
        pyg_graph = Data(
            x=torch.tensor(result['atom']).reshape(-1, 1),
            edge_index=torch.tensor(np.vstack([result['index1'], result['index2']])),
            edge_attr=torch.tensor(result['bond']), 
            y=torch.tensor([target], dtype=torch.float16)
        )
        record['id'] = p.stem
        record['structure'] = structure
        record['graph'] = result
        record['pyg_graph'] = pyg_graph
        cache.append(record)
    
    for p in tqdm(test_json_path):
        record = {}
        structure = read_pymatgen_dict(p)
        result = converter(structure)
        pyg_graph = Data(
            x=torch.tensor(result['atom']).reshape(-1, 1),
            edge_index=torch.tensor(np.vstack([result['index1'], result['index2']])),
            edge_attr=torch.tensor(result['bond'])
        )
        record['id'] = p.stem
        record['structure'] = structure
        record['graph'] = result
        record['pyg_graph'] = pyg_graph
        cache.append(record)
    
    cache = pd.DataFrame(cache)
    cache.to_pickle(Path(args.output_path)/f'data_cache_cutoff{args.cutoff}.pickle')
