"""
Test a trained autoencoder
"""
import argparse
import os
import random
import time

import cv2  # pytype: disable=import-error
import imgaug
import numpy as np
import torch as th

from pathlib import Path

from dalle_pytorch import DiscreteVAE
from dalle_pytorch.data_loader import preprocess_image, CheckFliplrPostProcessor, get_image_augmenter, denormalize

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs/recorded_data/")
parser.add_argument("-ae", "--ae-path", help="Path to saved AE", type=str, default="")
parser.add_argument("-n", "--n-samples", help="Max number of samples", type=int, default=20)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument("-augment", "--augment", action="store_true", default=False, help="Use image augmenter")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


if not args.folder.endswith("/"):
    args.folder += "/"

ae_path = Path(args.ae_path)
assert ae_path.exists(), 'AE model file does not exist'
assert not ae_path.is_dir(), \
    ('Cannot load VAE model from directory; please use a '
     'standard *.pt checkpoint. ')

loaded_obj = th.load(str(ae_path))
ae_params, weights = loaded_obj['hparams'], loaded_obj['weights']
autoencoder = DiscreteVAE(**ae_params).cuda()
autoencoder.load_state_dict(weights)

images = [im for im in os.listdir(args.folder) if im.endswith(".jpg")]
images = np.array(images)
n_samples = len(images)

augmenter = None
if args.augment:
    augmenter = get_image_augmenter()

# Small benchmark
start_time = time.time()
for _ in range(args.n_samples):
    # Load test image
    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image = cv2.imread(image_path)
    image = preprocess_image(image, convert_to_rgb=True)

    encoded = autoencoder.get_codebook_indices(th.as_tensor(image).unsqueeze(0).cuda())
    reconstructed_image = autoencoder.decode(encoded)

time_per_image = (time.time() - start_time) / args.n_samples
print(f"{time_per_image:.4f}s")
print(f"{1 / time_per_image:.4f}Hz")

errors = []

for _ in range(args.n_samples):
    # Load test image
    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image = cv2.imread(image_path)

    postprocessor = CheckFliplrPostProcessor()

    if augmenter is not None:
        input_image = augmenter.augment_image(image, hooks=imgaug.HooksImages(postprocessor=postprocessor))
    else:
        input_image = image

    if postprocessor.flipped:
        image = imgaug.augmenters.Fliplr(1).augment_image(image)

    cropped_image = preprocess_image(image, convert_to_rgb=True)
    cropped_image_show = preprocess_image(image, convert_to_rgb=False, normalize=False)
    encoded = autoencoder.get_codebook_indices(th.as_tensor(cropped_image).unsqueeze(0).cuda())
    reconstructed_image = autoencoder.decode(encoded).detach().cpu().numpy()
    reconstructed_image = denormalize(reconstructed_image)[0][:, :, ::-1]

    error = np.mean((cropped_image_show - reconstructed_image[0]) ** 2)
    errors.append(error)
    # Baselines error:
    # error = np.mean((cropped_image - np.zeros_like(cropped_image)) ** 2)
    # print("Error {:.2f}".format(error))

    # Plot reconstruction
    cv2.imshow("Original", image)
    # TODO: plot cropped and resized image
    cv2.imshow("Cropped", cropped_image_show)

    if augmenter is not None:
        cv2.imshow("Augmented", input_image)

    cv2.imshow("Reconstruction", reconstructed_image)
    # stop if escape is pressed
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

print(f"Min error: {np.min(errors):.2f}")
print(f"Max error: {np.max(errors):.2f}")
print(f"Mean error: {np.mean(errors):.2f} +/- {np.std(errors):.2f}")
print(f"Median error: {np.median(errors):.2f}")
