// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "utils/caseless.hpp"

#define USE_SMALL_VECTOR

#ifdef USE_SMALL_VECTOR
    #include "absl/container/inlined_vector.h"
#endif

namespace ov {
namespace intel_cpu {

using Dim = std::size_t;
#ifdef USE_SMALL_VECTOR
/// Inspired by LLVM SmallVector with ONNX Runtime adjustments for abseil.
/// https://github.com/llvm/llvm-project/blob/a85b37d0ca819776c6034c2dbda2b21e54e3393a/llvm/include/llvm/ADT/SmallVector.h#L1128-L1179
///
/// Helper class for calculating the default number of inline elements for
/// `InlinedVector<T>`.
/// This produces the following on MSVC x64
///    int8_t  -> 41
//     int16_t -> 21
//     int32_t -> 11
//     int64_t -> 6
//     std::string 40 -> 1
template <typename T>
struct CalculateInlinedVectorDefaultInlinedElements {
  // Parameter controlling the default number of inlined elements
  // for `InlinedVector<T>`.
  //
  // The default number of inlined elements ensures that
  // 1. There is at least one inlined element.
  // 2. `sizeof(InlinedVector<T>) <= kPreferredInlinedVectorSizeof` unless
  // it contradicts 1.
  static constexpr size_t kPreferredInlinedVectorSizeof = 64;

  // Largest allowed element size for default element count calculation.
  static constexpr size_t kElementSizeCutoff = 256;

  // static_assert that sizeof(T) is not "too big".
  //
  // Because the InlinedVector must have at least one inlined element, it is possible
  // for an arbitrarily large inlined element to allocate an arbitrarily large
  // amount of inline storage. So we want to call attention to these cases and
  // make sure that users are making an intentional decision if they request a lot of inline storage.
  //
  // We want this assertion to trigger in pathological cases, but otherwise
  // not be too easy to hit. To accomplish that, the cutoff is actually somewhat
  // larger than kPreferredInlinedVectorSizeof (otherwise,
  // `InlinedVector<InlinedVector<T>>` would be one easy way to trip it, and that
  // pattern seems useful in practice).
  //
  // One wrinkle is that this assertion is in theory non-portable, since
  // sizeof(absl::InlinedVector<T, 1>) is in general platform-dependent. However, we don't expect this
  // to be much of an issue, because most LLVM development happens on 64-bit
  // hosts, and therefore sizeof(T) is expected to *decrease* when compiled for
  // 32-bit hosts, dodging the issue. The reverse situation, where development
  // happens on a 32-bit host and then fails due to sizeof(T) *increasing* on a
  // 64-bit host, is expected to be very rare.
  static_assert(
      sizeof(T) <= kElementSizeCutoff,
      "You are trying to use a default number of inlined elements for "
      "`InlinedVector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `InlinedVector<T, N>` to make "
      "sure you really want that much inline storage.");

  // Discount the size of the header itself when calculating the maximum inline
  // bytes.
  static constexpr size_t InlinedVectorHeaderSize = sizeof(absl::InlinedVector<T, 1>) - sizeof(T);
  static constexpr size_t PreferredInlineBytes = kPreferredInlinedVectorSizeof - InlinedVectorHeaderSize;
  static constexpr size_t NumElementsThatFit = PreferredInlineBytes / sizeof(T);
  static constexpr size_t value =
      NumElementsThatFit == 0 ? 1 : NumElementsThatFit;
};

// Use InlinedVector for small arrays that can fit on a stack with a default
// value pre-calculated.
// Use TensorShapeVector for shapes.
template <typename T,
          size_t N = CalculateInlinedVectorDefaultInlinedElements<T>::value,
          typename Allocator = std::allocator<T>>
class InlinedVector : public absl::InlinedVector<T, N, Allocator> {
    using Super = absl::InlinedVector<T, N, Allocator>;

public:
    using Super::Super;
    InlinedVector(const std::vector<Dim>& vec) : InlinedVector(vec.begin(), vec.end()) {}

    InlinedVector& operator= (const std::vector<Dim>& lhs) {
        this->assign(lhs.begin(), lhs.end());
    }

    bool operator== (const std::vector<Dim>& lhs) const {
        if (this->size() != lhs.size())
            return false;

        return std::equal(this->begin(), this->end(), lhs.begin());
    }

    bool operator!= (const std::vector<Dim>& lhs) const {
        return !(this->operator=(lhs));
    }
};

template<typename T>
bool operator== (const std::vector<T>& rhs, const InlinedVector<T>& lhs) {
    if (rhs.size() != lhs.size())
        return false;

    return std::equal(rhs.begin(), rhs.end(), lhs.begin());
}

template<typename T>
bool operator!= (const std::vector<T>& rhs, const InlinedVector<T>& lhs) {
    return !(rhs == lhs);
}

using VectorDims = InlinedVector<Dim>;

#else
using VectorDims = std::vector<Dim>;
#endif

std::string dim2str(Dim dim);
std::string dims2str(const VectorDims& dims);

enum class Type {
    Unknown,
    If,
    Reorder,
    Input,
    Output,
    Eye,
    Convolution,
    Deconvolution,
    Lrn,
    Pooling,
    AdaptivePooling,
    FullyConnected,
    Softmax,
    Split,
    Concatenation,
    Eltwise,
    MatMul,
    Reshape,
    ShapeOf,
    NonZero,
    Tile,
    ROIAlign,
    ROIAlignRotated,
    ROIPooling,
    PSROIPooling,
    BatchToSpace,
    DepthToSpace,
    Pad,
    Transpose,
    SpaceToBatch,
    SpaceToDepth,
    StridedSlice,
    MemoryOutput,
    MemoryInput,
    RNNCell,
    RNNSeq,
    FakeQuantize,
    BinaryConvolution,
    DeformableConvolution,
    TensorIterator,
    Convert,
    ColorConvert,
    MVN,
    NormalizeL2,
    ScatterUpdate,
    ScatterElementsUpdate,
    ScatterNDUpdate,
    Interpolate,
    Reduce,
    Broadcast,
    EmbeddingBagPacked,
    EmbeddingBagOffsets,
    EmbeddingSegmentsSum,
    EmbeddingBagPackedSum,
    EmbeddingBagOffsetsSum,
    Gather,
    GatherElements,
    GatherND,
    GridSample,
    OneHot,
    RegionYolo,
    Roll,
    Reference,
    ShuffleChannels,
    DFT,
    RDFT,
    Math,
    CTCLoss,
    Bucketize,
    CTCGreedyDecoder,
    CTCGreedyDecoderSeqLen,
    CumSum,
    DetectionOutput,
    ExperimentalDetectronDetectionOutput,
    LogSoftmax,
    TopK,
    GatherTree,
    GRN,
    Range,
    Proposal,
    ReorgYolo,
    ReverseSequence,
    ExperimentalDetectronTopKROIs,
    ExperimentalDetectronROIFeatureExtractor,
    ExperimentalDetectronPriorGridGenerator,
    ExperimentalDetectronGenerateProposalsSingleImage,
    ExtractImagePatches,
    GenerateProposals,
    Inverse,
    NonMaxSuppression,
    MatrixNms,
    MulticlassNms,
    Multinomial,
    Subgraph,
    PriorBox,
    PriorBoxClustered,
    Interaction,
    MHA,
    RandomUniform,
    Unique,
    Ngram,
    ScaledDotProductAttention,
    PagedAttention,
    RoPE,
    CausalMaskPreprocess,
};

enum class Algorithm {
    Default,

    // Pooling algorithms
    PoolingMax,
    PoolingAvg,

    // Adaptive pooling algorithms
    AdaptivePoolingMax,
    AdaptivePoolingAvg,

    // Convolution algorithms
    ConvolutionCommon,
    ConvolutionGrouped,

    // Convolution algorithms
    DeconvolutionCommon,
    DeconvolutionGrouped,

    // Elementwise algorithms
    EltwiseAdd,
    EltwiseIsFinite,
    EltwiseIsInf,
    EltwiseIsNaN,
    EltwiseMultiply,
    EltwiseSubtract,
    EltwiseDivide,
    EltwiseFloor,
    EltwiseFloorMod,
    EltwiseMod,
    EltwiseMaximum,
    EltwiseMinimum,
    EltwiseSquaredDifference,
    EltwisePowerDynamic,
    EltwisePowerStatic,
    EltwiseMulAdd,
    EltwiseEqual,
    EltwiseNotEqual,
    EltwiseGreater,
    EltwiseGreaterEqual,
    EltwiseLess,
    EltwiseLessEqual,
    EltwiseLogicalAnd,
    EltwiseLogicalOr,
    EltwiseLogicalXor,
    EltwiseLogicalNot,
    EltwiseRelu,
    EltwiseGeluErf,
    EltwiseGeluTanh,
    EltwiseElu,
    EltwiseTanh,
    EltwiseSigmoid,
    EltwiseAbs,
    EltwiseSelect,
    EltwiseSqrt,
    EltwiseSoftRelu,
    EltwiseExp,
    EltwiseClamp,
    EltwiseSwish,
    EltwisePrelu,
    EltwiseMish,
    EltwiseHswish,
    EltwiseHsigmoid,
    EltwiseRoundHalfToEven,
    EltwiseRoundHalfAwayFromZero,
    EltwiseErf,
    EltwiseSoftSign,
    EltwiseLog,
    EltwiseBitwiseAnd,
    EltwiseBitwiseNot,
    EltwiseBitwiseOr,
    EltwiseBitwiseXor,

    // FakeQuantize algorithms
    FQCommon,
    FQQuantization,
    FQBinarization,

    // ROIPooling algorithms
    ROIPoolingMax,
    ROIPoolingBilinear,

    // ROIAlign algorithms
    ROIAlignMax,
    ROIAlignAvg,

    // PSROIPooling algorithms
    PSROIPoolingAverage,
    PSROIPoolingBilinear,
    PSROIPoolingBilinearDeformable,

    // Reduce algorithms
    ReduceL1,
    ReduceL2,
    ReduceAnd,
    ReduceOr,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceSumSquare,

    // Math algorithms
    MathAbs,
    MathAcos,
    MathAcosh,
    MathAsin,
    MathAsinh,
    MathAtan,
    MathAtanh,
    MathCeiling,
    MathCos,
    MathCosh,
    MathErf,
    MathFloor,
    MathHardSigmoid,
    MathNegative,
    MathReciprocal,
    MathSelu,
    MathSign,
    MathSin,
    MathSinh,
    MathSoftPlus,
    MathSoftsign,
    MathTan,
    // TensorIterator
    TensorIteratorCommon,
    TensorIteratorLoop,
    // Color conversions
    ColorConvertNV12toRGB,
    ColorConvertNV12toBGR,
    ColorConvertI420toRGB,
    ColorConvertI420toBGR,
};

extern const ov::intel_cpu::caseless_unordered_map<std::string, Type> type_to_name_tbl;

Type TypeFromName(const std::string& type);

std::string NameFromType(const Type type);

std::string algToString(const Algorithm alg);

}  // namespace intel_cpu
}  // namespace ov
