# Reproduction and evidence notes

The notebook's cells and stored outputs are retained unchanged. One metadata block was removed so that GitHub can render the file (see [below](#change-made-to-the-notebook)). The stored outputs describe a single historical execution on Google Colab; the original environment was not saved as a lockfile.

## Protocol as implemented

1. **Randomness:** cell 2 sets `torch`, NumPy and Python seeds to 0 and selects the CPU.
2. **Gallery:** cell 3 loads the CIFAR-10 test split through `datasets.load_dataset("cifar10", split="test")`. For each class, cell 4 resets the NumPy seed to 0 and draws 101 images without replacement: the first 100 join the gallery and the last becomes that class's query image.
3. **Image embeddings:** cell 6 encodes the 1,000 gallery images in batches of 50 with `openai/clip-vit-base-patch32`, resized to 224 × 224, and normalises them to unit length.
4. **Text embeddings:** cell 7 tokenises prompts with `CLIPProcessor` and normalises the text features. For an ensemble, cells 12 and 34 average the five template embeddings per class without re-normalising the average (see the issues below).
5. **Retrieval:** cells 8–13 rank all gallery images for each class-name query and record Recall@1 and Recall@5.
6. **Classification:** cell 34 assigns each gallery image the class with the highest similarity under each condition and computes accuracy, F1, top-k accuracy and confusion matrices.
7. **Linear probe:** cells 21–26 split the gallery into 80 training and 20 test images per class (the split reshuffles each class's indices with seed 0), fit `LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")` and compare it with zero-shot predictions on the same 200 test images.

Cell numbers are one-based positions in the notebook.

## Observed issues retained in the source

- **Recall@K is a hit-at-K check.** `compute_recall_at_k` returns 1 if any of the top K gallery labels matches the query class. With ten queries, every condition reached 1.0 and the Prompt Efficiency Index was 0. It is not recall over the 100 relevant images per class.
- **Ensemble averages are not re-normalised.** Averaging five unit vectors gives a vector slightly shorter than unit length, by an amount that depends on how similar the five prompts are. This does not change the image ranking for a single class query, so the retrieval results are unaffected. It does affect classification in cell 34, where an image is assigned the class with the largest dot product: classes whose five prompts agree more closely get a slightly longer vector and a small advantage. The canonical condition uses single, normalised prompts and is unaffected. The size of this effect was not measured, and it could be comparable to the small differences between conditions.
- **Query images are unused by the retrieval metric.** The one query image drawn per class supplies only its label; the queries are text prompts.
- **Mixed comparison in one cell.** Cell 28 prints zero-shot accuracy on all 1,000 gallery images (0.872) beside the probe's accuracy on 200 test images (0.915). Cell 24 makes the like-for-like comparison on the same 200 images (0.870 against 0.915). Cell 37's combined bar chart also places 1,000-image and 200-image results side by side.
- **Small differences.** The four conditions differ by at most 13 of 1,000 images, and there are no repeated samples or significance tests. The stored results do not establish a ranking of the prompt conditions.
- **Qualitative grids cover the ensembles only.** Cell 16 stores top-5 grids for Ensembles A, B and C, not for the canonical prompt.
- **Report wording.** The written report's abstract states that prompt formulation strongly affected retrieval accuracy. The stored outputs show saturated retrieval scores and classification accuracy between 86.2% and 87.5%.

These observations come from reading the source and stored outputs. They have not been resolved by changing the notebook.

## Change made to the notebook

The file in `notebooks/` differs from the submitted notebook only by the removal of `metadata.widgets`. That block held the state of Colab's download progress-bar widgets and lacked the `state` key that GitHub's notebook renderer requires, so the page would otherwise display as an invalid notebook. Every cell, output and execution count is unchanged, and the file is otherwise serialised exactly as submitted. Both hashes are recorded in [source-artifacts.json](source-artifacts.json).

## Dependencies

The stored outputs show Colab's Python 3.12. `requirements.txt` is a proposed compatibility set assembled from the notebook's imports and warnings. It has not been installed or runtime-tested for this repository.

- `transformers` is limited to versions before 4.52: the stored warning says the fast image processor becomes the default in 4.52, which may change outputs slightly.
- `scikit-learn` is limited to 1.5–1.6: the stored warning shows the `multi_class` argument used by the probe is deprecated from 1.5 and scheduled for removal.
- `datasets` downloads CIFAR-10 by the name `cifar10`; if that name stops resolving, the dataset is published as `uoft-cs/cifar10` on the Hugging Face Hub.

Rerunning on different library versions or hardware may change individual predictions. Compare any new run with the stored outputs rather than replacing them.

## Data and model attribution

CIFAR-10: Krizhevsky, A. (2009) *Learning Multiple Layers of Features from Tiny Images*. CLIP: Radford, A. et al. (2021) *Learning Transferable Visual Models From Natural Language Supervision*; weights from the `openai/clip-vit-base-patch32` model card on the Hugging Face Hub. Neither the dataset nor the weights are included in this repository.

## What was verified

The repository check validates Python syntax, notebook format, JSON and tracked documentation links. It does not execute the notebook or evaluate the model. File hashes confirm that the extracted figures come from the submitted notebook's stored outputs.
