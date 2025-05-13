# Train and prepare INT8 checkpoint for the AIU using Direct Quantization
This example builds on the [Direct Quantization (DQ) example](../DQ_SQ/README.md). We assume the user is already familiar with the DQ quantization process and would like to generate an INT8-quantized checkpoint that is made compliant with the requirements of the AIU.

Once created, this checkpoint can be run on the AIU by using an inference script from [aiu-fms-testing-utils](https://github.com/foundation-model-stack/aiu-fms-testing-utils).


## Requirements
- [FMS Model Optimizer requirements](../../README.md#requirements)

## QuickStart

**1. Prepare Data** as per DQ quantization process ([link](../DQ_SQ/README.md)). In this example, we assume the user wants to quantized RoBERTa-base model and has thus prepared the DQ data for it, stored under the folder `data_train` and `data_test`, by adapting the DQ example accordingly.

**2. Apply DQ with conversion** by providing the desired quantization parameters, as well as the flags `--save_ckpt_for_aiu` and `--recompute_narrow_weights`.

```bash
python  -m fms_mo.run_quant \
        --model_name_or_path "roberta-base" \
        --training_data_path data_train \
        --test_data_path data_test \
        --torch_dtype "float16" \
        --quant_method dq \
        --nbits_w 8 \
        --nbits_a 8 \
        --nbits_kvcache 32 \
        --qa_mode "pertokenmax"\
        --qw_mode "maxperCh" \
        --qmodel_calibration_new 1 \
        --output_dir "dq_test" \
        --save_ckpt_for_aiu \
        --recompute_narrow_weights
```
> [!TIP]
> - In this example, we are not evaluating the perplexity of the quantized model, but, if so desired, the user can add the `--eval_ppl` flag.
> - We set a single calibration example because the quantizers in use do not need calibration: weights remain static during DQ, so a single example will initialize the quantizer correctly, and the activation quantizer `pertokenmax` will dynamically recompute the quantization range at inference time, when running on the AIU.

**3. Reload checkpoint for testing**
