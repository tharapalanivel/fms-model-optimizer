#include <cublas_v2.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
// #include <c10/util/MaybeOwned.h>
// #include <ATen/TensorUtils.h>

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
using torch::Tensor;

template<typename T>
inline T* get_ptr(torch::Tensor& t)
{
    return reinterpret_cast<T*>(t.data_ptr());
}
inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}



// MatLayout indicates AT PYTHON LEVEL the layout of the input matrices
// At cublas level the operand positions are swapped before calling GEMM kernel
// i.e. MatLayout::NT calls cublas TN kernel, MatLayout TN calls cublas NT kernel
enum MatLayout {NN, NT, TN, TT};
// Templated function for different compute types
// Use cublas GEMM to compute act @ weight
// Here T is Compute Type and only 32F and 32I are supported (NOTICE: may not be the right setting for 16F)
template<typename T, MatLayout ML>
Tensor gemm(Tensor input_activations, Tensor weight)
{
    const auto is_act_transp    = (ML == MatLayout::TN) || (ML == MatLayout::TT);
    const auto is_weight_transp = (ML == MatLayout::NT) || (ML == MatLayout::TT);
    const auto compute_type = (std::is_same<T, int32_t>::value) ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
    const int m = input_activations.size((is_act_transp)?1:0);
    const int n = weight.size((is_weight_transp)?0:1);
    const int k = input_activations.size((is_act_transp)?0:1);
    cublasHandle_t       handle = at::cuda::getCurrentCUDABlasHandle();
    const at::ScalarType _st    = input_activations.scalar_type();
    CHECK_INPUT(input_activations, _st);
    CHECK_INPUT(weight, _st);
    TORCH_CHECK(_st == weight.scalar_type(), "Activation and weight must be the same type");
    TORCH_CHECK((compute_type==CUBLAS_COMPUTE_32I && _st == at::ScalarType::Char) ||
                (compute_type==CUBLAS_COMPUTE_32F && (_st == at::ScalarType::Half      ||
                                                    _st == at::ScalarType::BFloat16  ||
                                                    _st == at::ScalarType::Char      ||
                                                    _st == at::ScalarType::Float)), "Input type not supported");
    // At torch level (row-major) output is m x n
    const auto output_tensor_type = (compute_type==CUBLAS_COMPUTE_32I)                              ? at::ScalarType::Int   :
                                    (compute_type==CUBLAS_COMPUTE_32F && _st==at::ScalarType::Char) ? at::ScalarType::Float :
                                    _st;
    auto  output_tensor     = torch::empty({m, n}, torch::dtype(output_tensor_type).device(torch::kCUDA).requires_grad(false));
    cudaDataType_t cublas_type = _st == at::ScalarType::Half ? CUDA_R_16F :
                                _st == at::ScalarType::BFloat16 ? CUDA_R_16BF :
                                _st == at::ScalarType::Char ? CUDA_R_8I :
                                CUDA_R_32F;
    cudaDataType_t cublas_output_type = (compute_type==CUBLAS_COMPUTE_32I)                              ? CUDA_R_32I :
                                        (compute_type==CUBLAS_COMPUTE_32F && _st==at::ScalarType::Char) ? CUDA_R_32F:
                                        cublas_type;
    float alpha = 1.0f;
    float beta  = 0.0f;
    int32_t alpha_i32 = 1;
    int32_t beta_i32 = 0;
    auto alpha_ptr = (compute_type==CUBLAS_COMPUTE_32I) ? reinterpret_cast<const void*>(&alpha_i32) : reinterpret_cast<const void*>(&alpha);
    auto beta_ptr  = (compute_type==CUBLAS_COMPUTE_32I) ? reinterpret_cast<const void*>(&beta_i32)  : reinterpret_cast<const void*>(&beta);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    checkCublasStatus(cublasSetStream(handle, stream));
    checkCublasStatus(cublasGemmEx(handle,
                                        (is_weight_transp) ? CUBLAS_OP_T: CUBLAS_OP_N,
                                        (is_act_transp)    ? CUBLAS_OP_T: CUBLAS_OP_N,
                                        n,
                                        m,
                                        k,
                                        alpha_ptr,
                                        get_ptr<const void>(weight),
                                        cublas_type,
                                        (is_weight_transp) ? k:n,
                                        get_ptr<const void>(input_activations),
                                        cublas_type,
                                        (is_act_transp)    ? m:k,
                                        beta_ptr,
                                        get_ptr<void>(output_tensor),
                                        cublas_output_type,
                                        n,
                                        compute_type,
                                        CUBLAS_GEMM_DEFAULT));
    return output_tensor;
}
// Templated function for different compute types
// Use cublas GEMM to compute act @ weight
// T specifies scale type
// Compute type in this kernel is always 32I
template<typename T, MatLayout ML>
Tensor gemm_i8(Tensor input_activations, Tensor weight)
{
    const auto is_act_transp    = (ML == MatLayout::TN) || (ML == MatLayout::TT);
    const auto is_weight_transp = (ML == MatLayout::NT) || (ML == MatLayout::TT);
    const int m = input_activations.size((is_act_transp)?1:0);
    const int n = weight.size((is_weight_transp)?0:1);
    const int k = input_activations.size((is_act_transp)?0:1);
    cublasHandle_t   handle = at::cuda::getCurrentCUDABlasHandle();
    cublasLtHandle_t ltHandle = (cublasLtHandle_t)handle;

    const at::ScalarType _st    = input_activations.scalar_type();
    TORCH_CHECK(_st == at::ScalarType::Char, "gemm_i8 only support int8 inputs");
    CHECK_INPUT(input_activations, _st);
    CHECK_INPUT(weight, _st);
    const auto c_tensor_type = (std::is_same<T, int32_t>::value) ? at::ScalarType::Int : at::ScalarType::Char; // C
    auto  c_tensor     = torch::empty({m, n}, torch::dtype(c_tensor_type).device(torch::kCUDA).requires_grad(false));

    //// Seems to need bias vector for i8f32
    //const auto bias_tensor_type = (std::is_same<T, int32_t>::value) ? at::ScalarType::Int : at::ScalarType::Float; // Bias vector
    //auto  bias_tensor     = torch::empty({n}, torch::dtype(bias_tensor_type).device(torch::kCUDA).requires_grad(false));
    // Cublas doc: The accumulator value and the value from matrix C are typically converted to scale type before final scaling.
    // The value is then converted from scale type to the type of matrix D
    // At torch level (row-major) output is m x n
    const auto output_tensor_type = (std::is_same<T, int32_t>::value) ? at::ScalarType::Int : at::ScalarType::Char; // D
    auto  output_tensor     = torch::empty({m, n}, torch::dtype(output_tensor_type).device(torch::kCUDA).requires_grad(false));
    const auto compute_type = CUBLAS_COMPUTE_32I; // make sure we use int engine
    const auto cublas_type = CUDA_R_8I;           // A and B
    const auto cublas_c_type = (c_tensor_type == at::ScalarType::Int) ? CUDA_R_32I : CUDA_R_8I;
    const auto cublas_output_type = (output_tensor_type==at::ScalarType::Int) ? CUDA_R_32I : CUDA_R_8I; // D
    float alpha = 1.0f;
    float beta  = 0.0f;
    int32_t alpha_i32 = 1;
    int32_t beta_i32 = 0;
    auto alpha_ptr = (std::is_same<T, int32_t>::value) ? reinterpret_cast<const void*>(&alpha_i32) : reinterpret_cast<const void*>(&alpha);
    auto beta_ptr  = (std::is_same<T, int32_t>::value) ? reinterpret_cast<const void*>(&beta_i32)  : reinterpret_cast<const void*>(&beta);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    checkCublasStatus(cublasSetStream(handle, stream));
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc=NULL;
    cublasLtMatmulPreference_t preference = NULL;
    //int returnedResults                             = 0;
    //cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasOperation_t transa = (is_weight_transp) ? CUBLAS_OP_T: CUBLAS_OP_N;
    cublasOperation_t transb = (is_act_transp)    ? CUBLAS_OP_T: CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, compute_type, /*scaleType*/(std::is_same<T, int32_t>::value)?CUDA_R_32I:CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    //if (!std::is_same<T, int32_t>::value) {
    //    cublasLtEpilogue_t epilogue_type = CUBLASLT_EPILOGUE_BIAS;
    //    const void* bias_ptr = get_ptr<const void>(bias_tensor);
    //    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_type, sizeof(epilogue_type)));
    //    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    //}
    auto lda = (is_weight_transp) ? k:n;
    auto ldb = (is_act_transp)    ? m:k;
    auto ldc = n;
    auto ldd = n;
    checkCublasStatus(cublasLtMatrixLayoutCreate(
    &Adesc, cublas_type, transa == CUBLAS_OP_N ? n : k, transa == CUBLAS_OP_N ? k : n, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(
    &Bdesc, cublas_type, transb == CUBLAS_OP_N ? k : m, transb == CUBLAS_OP_N ? m : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, cublas_c_type, n, m, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, cublas_output_type, n, m, ldd));
    //checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    //checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
    //   preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, /*workspace*/nullptr, /*workspace_size*/0));
    //checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
    //   ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                        operationDesc,
                                        alpha_ptr,
                                        get_ptr<const void>(weight),
                                        Adesc,
                                        get_ptr<const void>(input_activations),
                                        Bdesc,
                                        beta_ptr,
                                        get_ptr<void>(c_tensor),
                                        Cdesc,
                                        get_ptr<void>(output_tensor),
                                        Ddesc,
                                        NULL, //&heuristicResult.algo,
                                        nullptr,
                                        0,
                                        stream));
    //if (preference)    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc)         checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)         checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)         checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)         checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    return output_tensor;
}
// batchedGemm
// reference: 1. https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp#L659C3-L659C3
//            2. https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp#L992
//            3. https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp#L422
template<typename T, MatLayout ML>
Tensor gemmBatched(Tensor m1, Tensor m2, Tensor output_tensor)
{   // in Python, m1.shape = [b,m,k], m2.shape = [b,k,n], out.shape = [b,m,n]
    //            we should pass the (m1, m2.T) in, and we call this layout "NT"
    //            Cuda will consider it as M1 (=m1.T=[b,k,m]) and M2 (=m2.T.T=[b,k,n]),
    //            so we call gemmBatched(M2 (flag as T), M1 (flag as N)),
    //            output should be [b,n,m] in C, then [b,m,n] when sent back to Py
    const auto is_m1_transp    = (ML == MatLayout::TN) || (ML == MatLayout::TT);
    const auto is_m2_transp = (ML == MatLayout::NT) || (ML == MatLayout::TT);
    const auto compute_type = (std::is_same<T, int32_t>::value) ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
    const int b = m1.size(0);
    const int m = m1.size((is_m1_transp)?2:1);
    const int n = m2.size((is_m2_transp)?1:2);
    const int k = m1.size((is_m1_transp)?1:2);
    cublasHandle_t       handle = at::cuda::getCurrentCUDABlasHandle();
    const at::ScalarType _st    = m1.scalar_type();
    CHECK_INPUT(m1, _st);
    CHECK_INPUT(m2, _st);
    TORCH_CHECK(_st == m2.scalar_type(), "Activation and weight must be the same type");
    TORCH_CHECK((compute_type==CUBLAS_COMPUTE_32I && _st == at::ScalarType::Char) ||
                (compute_type==CUBLAS_COMPUTE_32F && (_st == at::ScalarType::Half      ||
                                                    _st == at::ScalarType::BFloat16  ||
                                                    _st == at::ScalarType::Char      ||
                                                    _st == at::ScalarType::Float)), "Input type not supported");
    // At torch level (row-major) output is b x m x n
    const auto output_tensor_type = (compute_type==CUBLAS_COMPUTE_32I)                              ? at::ScalarType::Int   :
                                    (compute_type==CUBLAS_COMPUTE_32F && _st==at::ScalarType::Char) ? at::ScalarType::Float :
                                    _st;
    // auto  output_tensor     = torch::empty({b, m, n}, torch::dtype(output_tensor_type).device(torch::kCUDA).requires_grad(false));
    cudaDataType_t cublas_type = _st == at::ScalarType::Half ? CUDA_R_16F :
                                _st == at::ScalarType::BFloat16 ? CUDA_R_16BF :
                                _st == at::ScalarType::Char ? CUDA_R_8I :
                                CUDA_R_32F;
    cudaDataType_t cublas_output_type = (compute_type==CUBLAS_COMPUTE_32I)                              ? CUDA_R_32I :
                                        (compute_type==CUBLAS_COMPUTE_32F && _st==at::ScalarType::Char) ? CUDA_R_32F:
                                        cublas_type;
    float alpha = 1.0f;
    float beta  = 0.0f;
    int32_t alpha_i32 = 1;
    int32_t beta_i32 = 0;

    std::cout<< "Before calling gemmBatched" << std::endl;
    auto m1_ptr = (const void* const*)(m1.data_ptr());
    auto m2_ptr = (const void* const*)(m2.data_ptr());
    auto out_ptr = (void* const*)(output_tensor.data_ptr());
    std::cout<< "m1" << m1 << std::endl;
    std::cout<< output_tensor_type << std::endl << output_tensor << std::endl;

    std::cout<< "compute type is CUBLAS_COMPUTER_32I: " << (compute_type==CUBLAS_COMPUTE_32I) << std::endl;
    std::cout<< "cublas_type is CUDA_R_8I: " << (cublas_type==CUDA_R_8I) << std::endl;
    std::cout<< "cublas_output_type is CUDA_R_32I: " << (cublas_output_type==CUDA_R_32I) << std::endl;
    std::cout<< "output_tensor_type is at::ScalarType::Int: " << (output_tensor_type==at::ScalarType::Int) << std::endl;
    // std::cout<< torch::zeros({m, n}, torch::dtype(output_tensor_type).device(torch::kCUDA).requires_grad(false)) << std::endl;
    // -----

    auto alpha_ptr = (compute_type==CUBLAS_COMPUTE_32I) ? reinterpret_cast<const void*>(&alpha_i32) : reinterpret_cast<const void*>(&alpha);
    auto beta_ptr  = (compute_type==CUBLAS_COMPUTE_32I) ? reinterpret_cast<const void*>(&beta_i32)  : reinterpret_cast<const void*>(&beta);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    checkCublasStatus(cublasSetStream(handle, stream));
    checkCublasStatus(cublasGemmBatchedEx(handle,
                                        (is_m2_transp) ? CUBLAS_OP_T: CUBLAS_OP_N,
                                        (is_m1_transp) ? CUBLAS_OP_T: CUBLAS_OP_N,
                                        n,
                                        m,
                                        k,
                                        alpha_ptr,
                                        reinterpret_cast<const void* const*>(m2.data_ptr()),
                                        // m2_ptr,
                                        cublas_type,
                                        (is_m2_transp) ? k:n,
                                        reinterpret_cast<const void* const*>(m1.data_ptr()),
                                        // m1_ptr,
                                        cublas_type,
                                        (is_m1_transp) ? m:k,
                                        beta_ptr,
                                        reinterpret_cast<void* const*>(output_tensor.data_ptr()),
                                        // out_ptr,
                                        cublas_output_type,
                                        n,
                                        b,
                                        compute_type,
                                        CUBLAS_GEMM_DEFAULT));
    std::cout<< "After calling gemmBatched" << std::endl;
    std::cout<< m1 << std::endl;
    std::cout<< output_tensor_type << std::endl << output_tensor << std::endl;
    return output_tensor;
}

TORCH_LIBRARY(cublas_gemm, m)
{
    m.def("gemm_nn_f32",   gemm<float,         MatLayout::NN>);
    m.def("gemm_nt_f32",   gemm<float,         MatLayout::NT>);
    m.def("gemm_tn_f32",   gemm<float,         MatLayout::TN>);
    m.def("gemm_tt_f32",   gemm<float,         MatLayout::TT>);
    m.def("gemm_nt_i32",   gemm<int32_t,       MatLayout::NT>);
    m.def("gemm_nt_i8i32", gemm_i8<int32_t,    MatLayout::NT>);
    m.def("gemm_nt_i8i8",  gemm_i8<float,      MatLayout::NT>);
    m.def("gemmB_nt_i8i32", gemmBatched<int32_t,  MatLayout::NT>);
    m.def("gemmB_nt_i8f32", gemmBatched<float,    MatLayout::NT>);
}