## Patching Third Party Dependencies
Some dependencies clash with the current FMS Model Optimizer environment and we need to apply a patch.
To do this, we have provided a script in `fms-model-optimizer` named `install_patches.py`.

To run this script:
```
python3 install_patches.py
```

The following optional packages require a patch:
* `microxcaling`: Uses outdated versions of PyTorch-related packages