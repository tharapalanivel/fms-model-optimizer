# Model Optimization Using Post Training Quantization (PTQ)

FMS Model Optimizer supports [quantization](https://www.ibm.com/think/topics/quantization) of models which will enable the utilization of reduced-precision numerical format and specialized hardware to accelerate inference performance (i.e., make "calling a model" faster).


This is an example of [block sequential PTQ](https://arxiv.org/abs/2102.05426). Unlike quantization-aware training ([QAT](../QAT_INT8/README.md)) which simply trains the whole quantized model based on task loss, PTQ only trains one block at a time. Note that the "block" here could be a single layer, a transformer block, or a residual block. In this example we chose to use "transformer block" as it will provide better accuracy.  Furthermore, instead of using the task loss, PTQ relies on the MSE loss based on the differences between the original FP32 output and the quantized output of the block. The benefit of PTQ is that it requires much less computational resource and possibly shorter tuning time. One potential drawback is that the accuracy could be lower than that can be achieved by QAT, but in many cases PTQ can be comparable with QAT.


## Requirements

- [FMS Model Optimizer requirements](../../README.md#requirements)
- The inferencing step requires Nvidia GPUs with compute capability > 8.0 (A100 family or higher)
- NVIDIA cutlass package (Need to clone the source, not pip install). Preferably place in user's home directory: `cd ~ && git clone https://github.com/NVIDIA/cutlass.git`
- [Ninja](https://ninja-build.org/)
- `PyTorch 2.3.1` (as newer version will cause issue for the custom CUDA kernel)


## QuickStart

> [!NOTE]
> This example is based on the HuggingFace [Transformers Question answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering). Unlike our [QAT example](../QAT_INT8/README.md), which utilizes the training loop of the original code, our PTQ function will control the loop and the program will end before entering the original loop. Make sure the model doesn't get "tuned" twice!


There are **three main steps** to try out the example as follows:

#### **1.  Fine-tune a model** with 16-bit floating point (FP16) precision:

```shell
export CUDA_VISIBLE_DEVICES=0

python run_qa_no_trainer_ptq.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./fp16_ft_squad/ \
  --with_tracking \
  --report_to tensorboard \
  --attn_impl eager
```

> [!TIP]
> The script can take up to 20 mins to run (on a single A100). By default, it is configured for detailed logging. You can disable the logging by removing the `with_tracking` and `report_to` flags in the script.

#### **2.  Apply PTQ** on the fine-tuned model, which converts the precision data to 8-bit integer (INT8):

```shell
python run_qa_no_trainer_ptq.py \
  --model_name_or_path ./fp16_ft_squad \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --seed 0 \
  --do_ptq \
  --ptq_nbatch 128 \
  --ptq_batchsize 12 \
  --ptq_nouterloop 1000 \
  --ptq_coslr WA \
  --ptq_lrw 1e-05 \
  --ptq_lrcv_w 0.001 \
  --ptq_lrcv_a 0.001 \
  --output_dir ./ptq_on_fp16ft \
  --with_tracking \
  --report_to tensorboard
```

> [!TIP]
> The `model_name_or_path` from this section should match `output_dir` the previous section (step 1)

#### **3. Compare the accuracy and inference speed** of 16-bit floating point (FP16) and 8-bit integer (INT8) precision models:
> [!NOTE]
> All parameters are default, except for `batch size` and `do_lowering`

```shell
export TOKENIZERS_PARALLELISM=false

python run_qa_no_trainer_ptq.py \
  --model_name_or_path ./ptq_on_fp16ft \
  --dataset_name squad \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --attn_impl eager \
  --do_lowering
```

Checkout [Example Test Results](#example-test-results) to compare against your results.

## Example Test Results

The table below shows results obtained for the conditions listed:

|model|ptq_nbatch|Nouterloop|F1 score|PTQ tuning time (min.)|
|----|--:|---------:|----:|------------:|
|BERT|128|500       |81.5 |~10|
|    |128|1000      |85.08|~16|
|    |128|2000      |86.78|~25|
|    |128|3000      |87.63|~35|
|    |1000|2000     |86.82|~44|
|    |1000|3000     |87.50|~54|


`Nouterloop` and  `ptq_nbatch` are PTQ specific hyper-parameter.
Above experiments were run on v100 machine.

## Code Walk-through

In this section, we will deep dive into what happens during the example steps.

There are three parts to the example:

**1. Fine-tune a model with 16-bit floating point (FP16) precision**

Fine-tunes a BERT model on the question answering dataset, SQuAD. This step is based on the HuggingFace [Transformers Question answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering). It was modified to collect additional training information in case we would like to tweak the hyper-parameters later.

**2. Apply Quantization using PTQ**

For `INT8` quantization, we can achieve comparable accuracy with `FP16` by using [quantization-aware training (QAT)](https://arxiv.org/abs/2407.11062) or [post-training quantization (PTQ)](https://arxiv.org/abs/2102.05426) techniques. In this example we use PTQ.

In a nutshell, PTQ simply quantizes the weight and activation tensors in a block sequential manner, at each block optimizes for quantization errors. (i.e. quantization and optimization together happen block by block / one block at a time starting from 1st in a sequential manner)

```python
from fms_mo import qmodel_prep, qconfig_init

# Create a config dict using a default recipe and CLI args
# If same item exists in both, args take precedence over recipe.
qcfg = qconfig_init(recipe = 'ptq_int8', args=args)
qcfg["tb_writer"] = accelerator.get_tracker("tensorboard", unwrap=True)
qcfg["loader.batchsize"] = args.per_device_train_batch_size


# Prepare a list of "ready-to-run" data for calibration
exam_inp = [{k:v for k,v in next(iter(train_dataloader)).items() if 'position' not in k}
        for _ in range(qcfg['qmodel_calibration']) ]

ptq_mod_candidates = list( model.bert.encoder.layer )
qmodel_prep(model, exam_inp, qcfg, optimizer, use_dynamo = True)
calib_PTQ_lm(qcfg, model, train_dataloader, ptq_mod_candidates)

logger.info(f"--- Accuracy of {args.model_name_or_path} before QAT/PTQ")
```

**3. Evaluate Inference Accuracy and Speed**

> [!NOTE]
> This step will compile an external kernel for INT matmul, which currently only works with `PyTorch 2.3.1`.

Here is an example code snippet used for evaluation:

```python
from fms_mo.modules.linear import QLinear, QLinearINT8Deploy
# ...

# Only need 1 batch (not a list) this time, will be used by `torch.compile` as well.
exam_inp = next(iter(train_dataloader))

qcfg = qconfig_init(recipe = 'qat_int8', args=args)
qcfg['qmodel_calibration'] = 0 # <----------- NOTE 1
qmodel_prep(model, exam_inp, qcfg, optimizer, use_dynamo = True,
            ckpt_reload=args.model_name_or_path) # <----------- NOTE 2

# ----------- NOTE 3
mod2swap = [n for n,m in model.named_modules() if isinstance(m, QLinear)]
for name in mod2swap:
    parent_name, module_name = _parent_name(name)
    parent_mod = model.get_submodule(parent_name)
    qmod = getattr(parent_mod, module_name)
    setattr(parent_mod, module_name, QLinearINT8Deploy.from_fms_mo(qmod))

# ...

with torch.no_grad():
    model = torch.compile(model) #, mode='reduce-overhead') # <----- NOTE 4
    model(**exam_inp)

# ...

return # Stop the run here, no further training loop
```
