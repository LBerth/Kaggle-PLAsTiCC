import numpy as np
import pandas as pd
import tqdm

meta_data = pd.read_csv('../Data/training_set_metadata.csv')
test_meta_data = pd.read_csv('../Data/test_set_metadata.csv')

targets = np.hstack([np.unique(meta_data['target']), [99]])
target_map = {j:i for i, j in enumerate(targets)}
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids

# Build the flat probability arrays for both the galactic and extragalactic groups
galactic_cut = meta_data['hostgal_specz'] == 0
galactic_data = meta_data[galactic_cut]
extragalactic_data = meta_data[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])

# Add class 99 (id=14) to both groups.
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)

galactic_probabilities = np.zeros(15)
galactic_probabilities[galactic_classes] = 1. / len(galactic_classes)
extragalactic_probabilities = np.zeros(15)
extragalactic_probabilities[extragalactic_classes] = 1. / len(extragalactic_classes)

# Apply this prediction to a table
def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    return np.array(probs)

pred = do_prediction(meta_data)
test_pred = do_prediction(test_meta_data)

# Il faut maintenant écrire les prédictions dans un fichier submission
