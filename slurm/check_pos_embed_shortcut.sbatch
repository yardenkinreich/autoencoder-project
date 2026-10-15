#!/bin/bash
#SBATCH --job-name=check-pos-embed-shortcut
#SBATCH --partition=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm-%x-%j.out

# Decisive version of check_pos_embed_paths.sbatch: same two loading paths
# (stock's dynamic img_size=518 interpolate_pos_encoding() vs every
# trained/frozen checkpoint's _resize_pretrained_pos_embed() baked in at
# img_size=224), but embedding the REAL new_set_final_test craters and
# running the actual PC1-vs-brightness diagnostic on each, instead of a
# cosine-similarity proxy on random noise. If switching ONLY the pos_embed
# path (all other weights held identical - both loaded from the exact same
# stock state_dict) reproduces stock's non-collapsed result (PC1~34%,
# r~-0.47) vs diet_cp_frozen's collapsed one (PC1~98%, r~1.00), that
# isolates the pos_embed handling itself as the cause, independent of any
# training recipe tested today.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

PYTHONPATH=$(pwd) pixi run -e default python -c "
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.train.ssl_meta_arch import _resize_pretrained_pos_embed
from src.eval import pipeline as P
from src.eval import metrics as M

device = 'cuda'
stock_sd = torch.load('data/raw/pretrained/dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=False)

model_a = vit_small(img_size=518, patch_size=14, in_chans=3, block_chunks=0, init_values=1.0)
model_a.load_state_dict(stock_sd, strict=True)
model_a.to(device).eval()

sd_b = {k: v.clone() for k, v in stock_sd.items()}
model_b = vit_small(img_size=224, patch_size=14, in_chans=3, block_chunks=0, init_values=1.0)
_resize_pretrained_pos_embed(sd_b, model_b.pos_embed)
model_b.load_state_dict(sd_b, strict=True)
model_b.to(device).eval()

# real new_set_final_test craters, same protocol diagnose_shortcut.py uses
metadata_csv = 'data/processed_wac_100m_new/sigma/100/new_test_set/metadata.csv'
labels_csv = 'configs/correct_crater_with_labels_final.csv'
dat_path = 'data/processed_wac_100m_new/sigma/100/new_test_set/craters.dat'
imgs_dir = 'data/processed_wac_100m_new/sigma/100/new_test_set/crater_crops'
class_names = ['v-fr', 'r-fr', 'r-dr', 'v-dr']
split_df = pd.read_csv('configs/new_test_set_split.csv', dtype={'crater_id': str})
allow = set(split_df.loc[split_df['split'] == 'final_test', 'crater_id'])
labels = P.labels_from_memmap_csv(metadata_csv, labels_csv, class_names, id_allowlist=allow)

from src.eval.holdout import load_holdout_dat
import torch.nn.functional as F
data = load_holdout_dat(dat_path)
row_idx = labels['row_idx'].to_numpy()
imgs = torch.from_numpy(np.asarray(data[row_idx])).float()  # (N,1,H,W) in [0,1]
imgs224 = F.interpolate(imgs, size=(224, 224), mode='bicubic', align_corners=False)
imgs224 = imgs224.repeat(1, 3, 1, 1).to(device)

def embed(model):
    outs = []
    with torch.no_grad():
        for i in range(0, len(imgs224), 64):
            batch = imgs224[i:i+64]
            out = model(batch, is_training=True)['x_norm_clstoken']
            out = F.normalize(out, dim=1, p=2)
            outs.append(out.cpu().numpy())
    return np.concatenate(outs, axis=0)

lat_a = embed(model_a)
lat_b = embed(model_b)

feats = M.image_intensity_features(labels, imgs_dir)
merged = labels[['crater_id']].merge(feats, on='crater_id', how='inner')
brightness = merged['brightness'].to_numpy()

for name, lat in [('A: stock (518 native, dynamic interp)', lat_a),
                  ('B: 224-baked (_resize_pretrained_pos_embed)', lat_b)]:
    pca = PCA(n_components=min(10, lat.shape[1]), random_state=0)
    scores = pca.fit_transform(lat)
    pc1 = scores[:, 0]
    ev = pca.explained_variance_ratio_[0]
    r, p = pearsonr(pc1, brightness)
    rs, ps = spearmanr(pc1, brightness)
    print(f'{name}: PC1 explained={ev:.3f}  pearson_r={r:.3f} (p={p:.2e})  spearman_r={rs:.3f}')
"
echo "EXIT=$?"
