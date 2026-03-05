# Neural Compression Distillation (Standalone Notebook)

`neuralcompression_distill.ipynb` is a fully self-contained Colab notebook that trains a CNN teacher on CIFAR-100, generates an offline per-layer distillation dataset, and trains multiple student models to match teacher layer outputs with a one-hot layer ID conditioning channel.

## What the notebook does

1. **Defines the teacher CNN**
   - A multi-block CNN with fixed spatial shape across all layers.
   - Uses CIFAR-100 classification to train the teacher.

2. **Trains the teacher**
   - Standard data augmentation for CIFAR-100.
   - Saves the best checkpoint by validation accuracy.

3. **Generates the offline layer dataset**
   - Runs the frozen teacher over the training set.
   - Captures `(input_to_layer_l, output_of_layer_l)` for each layer.
   - Writes per-layer tensors to disk and a manifest with shapes.

4. **Defines the student CNN**
   - Smaller CNN that receives a padded teacher-layer input plus a one-hot layer ID (extra channels).
   - Outputs full teacher layer activations.

5. **Trains multiple student variants**
   - Targets parameter ratios: 10%, 20%, 30%, 40%, 50% of teacher params.
   - Uses a masked MSE to ignore padded channels and spatial regions.
   - Balanced sampling so all layers contribute equally.

## Inputs and outputs

- **Student input shape**: `[B, C_in + L, H, W]`
  - `C_in`: max teacher input channels across layers (padded).
  - `L`: number of teacher layers (one-hot layer ID channels).
  - `H, W`: CIFAR-100 spatial size (32x32).

- **Student output shape**: `[B, max_out_channels, H, W]`
  - Masked to score only valid teacher channels per layer.

## Where data is stored

- CIFAR-100 is downloaded automatically by torchvision with `download=True`.
- The default dataset root is `~/.torch/datasets` (set in the notebook).

## Running the notebook

Open `neuralcompression_distill.ipynb` and execute the cells in order:

1. Train the teacher
2. Generate the layer dataset
3. Train student variants

## Notes

- The teacher keeps the same spatial shape across all layers to keep student input/output shapes consistent.
- The distillation dataset is generated once and reused for all student variants.
- The notebook is standalone: no external project files are required.
# neuralcompression
