# Finding point sources in a dirty image

Interactive demo of my BSc (Honours) project at Rhodes University (2021), supervised by Dane Brown.

A linear SVM sweeps a simulated MeerKAT dirty image tile by tile. It boxes every source it finds and labels each one a point source or extended emission, with a confidence score. Everything runs in the browser.

**Live demo:** open `index.html` through GitHub Pages.

## What's in it

- `index.html`, `core.js`: the page and the detector (background mesh, peak finding, linear SVM, all in JavaScript)
- `data.json`: the Experiment 3 unseen dirty image (1344 × 1344, 1,000 point sources and 100 extended sources) with ground truth
- `model.json`: the trained linear SVM (C = 10) with its scaling folded into the weights
- `sky.png`, `es_outline.png`: the true-sky layer and extended-source outlines

## How it works

1. The image is split into tiles. Each tile's brightest local peak is kept if it rises above a SExtractor-style background mesh by the chosen threshold.
2. A 51 × 51 cutout around the peak goes to the linear SVM, which scores it as a point source or not.
3. Ground truth, rebuilt from the simulation's random seed, marks each verdict right or wrong.

The SVM was retrained with the thesis settings on the Experiment 3 training cutouts, using the training scaler for unseen data. It scores 89.5% on all 1,275 unseen cutouts.
