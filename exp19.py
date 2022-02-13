from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import argparse
import os
from sklearn.model_selection import KFold
from monty.serialization import dumpfn, loadfn
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from pymatgen.core import Structure
from megnet.data.crystal import CrystalGraph
from megnet_models import MEGNetModel


def energy_within_threshold(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)


def scheduler(epoch, lr):
    if epoch < 60:
        return LR
    elif 60 <= epoch < 120:
        return LR / 5
    else:
        return LR / 10


if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--inference', action='store_true', help='skip training')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    # %%
    NAME = 'exp_19'
    SEED = 2022
    CV = 5
    MAX_EPOCHS = 180
    BATCH_SIZE = 128
    LR = 1e-3

    # Use pre-defined graphs
    CUTOFF = 5
    GAUSS_CENTER = np.linspace(0, CUTOFF + 1, 100)
    GAUSS_WIDTH = 0.5 # kore

    EXTERNAL = None

    # %%
    INPUT_DIR = Path('data')
    OUTPUT_DIR = Path('results') / NAME
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # %%
    cache = pd.read_pickle('data/data_cache_cutoff4.pickle')
    train = pd.read_csv('data/dichalcogenides_public/targets.csv')
    if EXTERNAL is not None:
        external = pd.read_pickle(EXTERNAL)

    # %%
    splitter = KFold(n_splits=CV, shuffle=True, random_state=SEED)
    fold_iter = list(splitter.split(X=train, y=train['band_gap']))

    # %%
    results = []
    test_ids = [ p.stem for p in (INPUT_DIR/'dichalcogenides_private/structures/').glob('*.json') ]
    test_structures = [ cache.loc[cache['id'] == _id, 'structure'].values[0] for _id in test_ids ]
    test_predictions = pd.DataFrame.from_dict({'id': test_ids}, orient='columns')
    outoffolds = np.zeros((len(train), 1), dtype=float)
    train_all = train.merge(cache, left_on='_id', right_on='id', how='left')

    if not args.inference: # Training and inference
        model = MEGNetModel(
            graph_converter=CrystalGraph(cutoff=CUTOFF),
            centers=GAUSS_CENTER,
            width=GAUSS_WIDTH,
            loss="mae",
            npass=2,
            lr=LR,
            metrics=energy_within_threshold, 
            metrics_mode='max'
        )
        model.load_weights('data/efermi.hdf5')
        res = model.train(
            train_all['structure'],
            train_all['band_gap'],
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            dirname=str(OUTPUT_DIR),
            filename=f'checkpoint',
            verbose=2,
            callbacks=[
                LearningRateScheduler(scheduler),
                ModelCheckpoint(
                    str(OUTPUT_DIR/'checkpoint.hdf5'), 
                    monitor='energy_within_threshold', 
                    save_best_only=True, 
                    mode='max'
                )
            ],
        )
        results = pd.DataFrame({
            'loss': res.history.history['loss'],
            'metric': res.history.history['energy_within_threshold']
        })
        results['epoch'] = np.arange(len(results)) + 1
        results['best_epoch'] = results['metric'].argmax() + 1
        results['best_metric'] = results['metric'].max()
        results[[
            'epoch', 'best_epoch', 
            'loss', 'metric', 'best_metric']].to_csv(OUTPUT_DIR/f'results.csv', float_format="%10.5f", index=False)

        model.load_weights(str(OUTPUT_DIR/'checkpoint.hdf5'))
        test_predictions['predictions'] = model.predict_structures(test_structures)
        outoffolds = model.predict_structures(train_all['structure'].tolist())

    else: # inference only
        model = MEGNetModel(
            graph_converter=CrystalGraph(cutoff=CUTOFF),
            centers=GAUSS_CENTER,
            width=GAUSS_WIDTH,
            loss="mae",
            npass=2,
            lr=LR,
            metrics=energy_within_threshold, 
            metrics_mode='max'
        )
        model.load_weights(str(OUTPUT_DIR/'checkpoint.hdf5'))
        test_predictions['predictions'] = model.predict_structures(test_structures)
        outoffolds = model.predict_structures(train_all['structure'].tolist())

    test_predictions[['id', 'predictions']].to_csv(OUTPUT_DIR/'submission.csv', index=False)
    train['outoffolds'] = outoffolds
    train[['_id', 'outoffolds']].to_csv(OUTPUT_DIR/'outoffolds.csv', index=False)
