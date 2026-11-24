# HalluGuard

HalluGuard checks whether a source passage supports an LLM-generated claim. A fine-tuned DeBERTa-v3 Large cross-encoder reads the passage and the claim together, then returns a verdict and a support score.

[Demo notebook](demo/HalluGuard_Model_Showcase.ipynb) · [Sample pairs](examples/sample_pairs.csv) · [Research paper](docs/HalluGuard_Final_Research_Paper.pdf)

## Try an example

Use the same passage for both claims and change only the year.

| Source passage | Claim | Expected judgment |
| --- | --- | --- |
| Napoleon was defeated at Waterloo in 1815. | Napoleon was defeated at Waterloo in 1815. | Supported |
| Napoleon was defeated at Waterloo in 1815. | Napoleon was defeated at Waterloo in 1820. | Unsupported |

The notebook runs six examples, displays expected and predicted judgments side by side, and lets you enter a passage and claim of your own.

### Run on Kaggle

1. Download [the demo notebook](demo/HalluGuard_Model_Showcase.ipynb) and import it into a new Kaggle notebook.
2. Under **Input → Add Input → Models**, attach the HalluGuard model version shared by the project owner. Your account needs access to the selected version.
3. Run the setup cell, then run the sample table or edit the custom passage and claim. Keep `EXPORT_MODEL = False` for ordinary testing.

### Run locally

With Python and Jupyter, install the notebook dependencies and start Jupyter from the repository root.

```shell
python -m pip install -r requirements.txt
python -m notebook demo/HalluGuard_Model_Showcase.ipynb
```

Download the trained model export from Kaggle and set `MODEL_DIR` in the setup cell to the directory containing `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `threshold_config.json`. The notebook uses a CUDA GPU when available and falls back to CPU.

## How it works

1. Tokenize the source passage first and the claim second, with a 256-token limit for the pair.
2. Run the pair through the two-label DeBERTa-v3 Large checkpoint.
3. Return **Supported** when the support score reaches the cutoff in the saved model config. The archived export uses **0.42**. **Unsupported** covers both contradiction and insufficient evidence.

## Recorded evaluation

The research paper reports these results on the archived **32,602-pair** test set. Both rows use the same checkpoint and test scores.

| Support cutoff | Accuracy | Supported precision | Supported recall | Supported F1 |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 86.09% | 79.55% | 80.36% | 79.95% |
| 0.42 | 85.28% | 76.48% | 82.81% | 79.52% |

Lowering the cutoff recovered 276 additional supported pairs and accepted 541 additional unsupported pairs. The paper documents passage reuse across the development and test exports and examples of reference-label errors. Performance on independently labeled claims from new sources remains unmeasured.

## Repository contents

| Path | What's inside |
| --- | --- |
| [`demo/`](demo/) | Inference notebook with six examples and a custom-input cell |
| [`examples/`](examples/) | Hand-written sample pairs with expected labels |
| [`training/`](training/) | Training and evaluation code from the Kaggle notebook |
| [`docs/`](docs/) | Final two-column research paper |

The training notebook expects the original development and test CSVs as Kaggle inputs. This release includes hand-written samples for inference. The trained weights are distributed separately through Kaggle.

## Model provenance

HalluGuard starts from [Moritz Laurer's DeBERTa-v3 Large NLI checkpoint](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) and fine-tunes it for binary evidence support. The [paper](docs/HalluGuard_Final_Research_Paper.pdf) contains the experimental configuration and references.
