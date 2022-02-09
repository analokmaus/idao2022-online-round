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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pymatgen.core import Structure
from megnet.data.crystal import CrystalGraph, CrystalGraphWithBondTypes
from megnet_models import MEGNetModel


def energy_within_threshold(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)


if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    # %%
    NAME = 'exp_7'
    SEED = 2022
    CV = 5
    MAX_EPOCHS = 100
    BATCH_SIZE = 128
    LR = 1e-4 

    CUTOFF = 4
    GAUSS_CENTER = np.linspace(0, CUTOFF + 1, 100)
    GAUSS_WIDTH = 0.5

    # %%
    INPUT_DIR = Path('data')
    OUTPUT_DIR = Path('results') / NAME
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # %%
    cache = pd.read_pickle('data/data_cache_cutoff4.pickle')
    train = pd.read_csv('data/dichalcogenides_public/targets.csv')

    # %%
    splitter = KFold(n_splits=CV, shuffle=True, random_state=SEED)
    fold_iter = list(splitter.split(X=train, y=train['band_gap']))


    # %%
    results = []
    test_ids = [ p.stem for p in (INPUT_DIR/'dichalcogenides_private/structures/').glob('*.json') ]
    test_structures = [ cache.loc[cache['id'] == _id, 'structure'].values[0] for _id in test_ids ]
    test_predictions = pd.DataFrame.from_dict({'id': test_ids}, orient='columns')
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        train_fold, valid_fold = train.iloc[train_idx], train.iloc[valid_idx]
        train_fold = train_fold.merge(cache, left_on='_id', right_on='id', how='left')
        valid_fold = valid_fold.merge(cache, left_on='_id', right_on='id', how='left')

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
        model.load_weights('data/band_gap_regression.hdf5')
        res = model.train(
            train_fold['structure'],
            train_fold['band_gap'],
            validation_structures=valid_fold['structure'],
            validation_targets=valid_fold['band_gap'],
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            dirname=str(OUTPUT_DIR),
            filename=f'fold_{fold}',
            verbose=2,
            # callbacks=[
            #     ReduceLROnPlateau(
            #         monitor='loss', factor=0.5, patience=5, mode='min', min_lr=1e-5, min_delta=1e-6)],
            # automatic_correction=True,
        )

        result_fold = pd.DataFrame(loadfn(OUTPUT_DIR/f'fold_{fold}.json'))
        result_fold = pd.DataFrame(loadfn(OUTPUT_DIR/f'fold_{fold}.json'))
        result_fold['trn_loss'] = res.history.history['loss']
        result_fold['trn_energy_within_threshold'] = res.history.history['energy_within_threshold']
        result_fold[
            ['epoch', 'best_epoch', 
            'trn_loss', 'trn_energy_within_threshold', 
            'val_mae', 'val_energy_within_threshold', 'best_energy_within_threshold']].to_csv(OUTPUT_DIR/f'fold_{fold}.csv', float_format="%10.5f", index=False)
        results.append(result_fold.iloc[-1])

        model.load_weights(str(OUTPUT_DIR/f'fold_{fold}.hdf5'))
        test_predictions[f'fold_{fold}'] = model.predict_structures(test_structures)

    results = pd.concat(results, axis=1)
    results.columns = np.arange(CV)
    results.to_csv(OUTPUT_DIR/'summary.csv', float_format="%10.5f")

    test_predictions['predictions'] = test_predictions[[ f'fold_{i}' for i in range(CV) ]].mean(1)
    test_predictions[['id', 'predictions']].to_csv(OUTPUT_DIR/'submission.csv', index=False)
