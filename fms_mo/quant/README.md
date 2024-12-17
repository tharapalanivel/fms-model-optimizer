# Notice

In the `ptq.py` file in this folder, Class `StraightThrough`, function `_fold_bn`, `fold_bn_into_conv`, `reset_bn`, and `search_fold_and_remove_bn` are modified from `QDROP` repository on GitHub. 
For the original code, see [QDrop](https://github.com/wimh966/QDrop/tree/qdrop/qdrop/quantization) which has no license stipulated.

In the `quantizers.py` file in this folder, Class/function `MSEObserver`, `ObserverBase`, `fake_quantize_per_channel_affine`, `fake_quantize_per_tensor_affine`, `_transform_to_ch_axis`, `CyclicTempDecay`, `LinearTempDecay`, `AdaRoundSTE`, `AdaRoundQuantizerare` are modified from `BRECQ` repository on GitHub. For the original code, see [BRECQ](https://github.com/yhhhli/BRECQ) with the following license.


```license
MIT License

Copyright (c) 2021 Yuhang Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```