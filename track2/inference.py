from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from pymatgen.core import Structure
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel
import gc


CUTOFF = 5
GAUSS_CENTER = np.linspace(0, CUTOFF + 1, 100)
GAUSS_WIDTH = 0.5

INPUT_DIR = Path('data')
OUTPUT_DIR = Path('results')


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)


if __name__ == '__main__':

    test = {p.stem:read_pymatgen_dict(p) for p in (INPUT_DIR/'dichalcogenides_private/structures/').glob('*.json') }
    test_predictions = pd.DataFrame({'id': test.keys(), 'structures': test.values()})
    weight_paths = list(OUTPUT_DIR.glob('*.hdf5'))

    for fold, wp in enumerate(weight_paths):
        model = MEGNetModel(
            graph_converter=CrystalGraph(cutoff=CUTOFF),
            centers=GAUSS_CENTER,
            width=GAUSS_WIDTH,
            npass=2
        )
        model.load_weights(str(wp))
        test_predictions[f'fold_{fold}'] = model.predict_structures(test_predictions['structures'])
        print(f'fold{fold} done')
        del model; gc.collect()

    test_predictions['predictions'] = test_predictions[[ f'fold_{i}' for i in range(len(weight_paths)) ]].mean(1)
    test_predictions[['id', 'predictions']].to_csv('submission.csv', index=False)
