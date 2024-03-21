// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update.h"

#include "common/cpu_memcpy.h"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset12.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace dnnl;

#ifdef NDEBUG
#define ASSERT_DEBUG_ONLY(...)
#else
#define ASSERT_DEBUG_ONLY(...) OPENVINO_ASSERT(__VA_ARGS__)
#endif

namespace ov {
namespace intel_cpu {
namespace node {

bool ScatterUpdate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto scatterElemUpd3 = ov::as_type_ptr<const ov::opset3::ScatterElementsUpdate>(op);
        auto scatterElemUpd12 = ov::as_type_ptr<const ov::opset12::ScatterElementsUpdate>(op);
        auto scatterUpd = ov::as_type_ptr<const ov::opset3::ScatterUpdate>(op);
        auto scatterNdUpd = ov::as_type_ptr<const ov::opset4::ScatterNDUpdate>(op);
        if (!scatterElemUpd3 && !scatterElemUpd12 && !scatterUpd && !scatterNdUpd) {
            const std::string opType = op->get_type_name();
            errorMessage = std::string("Type ") + opType + " is not supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool ScatterUpdate::isExecutable() const {
    return !isInputTensorAtPortEmpty(DATA_ID);
}

ScatterUpdate::ScatterUpdate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)),
          dataSize(0lu), indicesSize(0lu), axisSize(0lu),
          dataPrec(ov::element::undefined),
          indicesPrec(ov::element::undefined),
          axisPrec(ov::element::undefined) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = std::string(op->get_type_name()) + " node with name '" + getName() + "'";
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v12::ScatterElementsUpdate>(op);
    if (node) {
        reduction_type = node->get_reduction();
        use_init_val = node->get_use_init_val();
    } else {
        reduction_type = ov::op::v12::ScatterElementsUpdate::Reduction::NONE;
    }
}

void ScatterUpdate::getSupportedDescriptors() {
    if ((getParentEdges().size() != 3) && (getParentEdges().size() != 4))
        OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");

    if (getInputShapeAtPort(DATA_ID).getRank() < 1 ||
        getInputShapeAtPort(INDICES_ID).getRank() < 1 ||
            getInputShapeAtPort(UPDATE_ID).getRank() < 1) {
        OPENVINO_THROW(errorPrefix, " do not support scalar input");
    }

    Type scatterUpdateType = getType();
    if (scatterUpdateType == Type::ScatterUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
        axisRelaxed = true;
    } else if (scatterUpdateType == Type::ScatterElementsUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterElementsUpdate;
        axisRelaxed = true;
    } else if (scatterUpdateType == Type::ScatterNDUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterNDUpdate;
        axisRelaxed = false;
    } else {
        OPENVINO_THROW(errorPrefix, " is not supported");
    }
}

void ScatterUpdate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& srcDataDim = getInputShapeAtPort(DATA_ID).getDims();
    const auto& indicesDim = getInputShapeAtPort(INDICES_ID).getDims();
    const auto& updateDim =  getInputShapeAtPort(UPDATE_ID).getDims();
    const auto& dstDataDim = getOutputShapeAtPort(0).getDims();

    size_t srcRank = srcDataDim.size();
    size_t indicesRank = indicesDim.size();
    size_t updateRank = updateDim.size();
    size_t dstRank = dstDataDim.size();

    // common check
    if (srcRank != dstRank) {
        OPENVINO_THROW(errorPrefix, " should have same rank for input and output tensor");
    } else {
        for (size_t r = 0; r < srcRank; r++) {
            if (!dimsEqualWeak(srcDataDim[r], dstDataDim[r])) {
                OPENVINO_THROW(errorPrefix,
                               " should have same shape for input and output tensor. The input shape is ",
                               srcDataDim[r],
                               ", while output shape is ",
                               dstDataDim[r],
                               " for ",
                               r,
                               "th dimension");
            }
        }
    }
    // specific check
    switch (scatterUpdateMode) {
        case ScatterUpdateMode::ScatterUpdate: {
            if (updateRank != (srcRank + indicesRank - 1)) {
                OPENVINO_THROW(errorPrefix,
                               " do not have matched tensor rank relationship for input, indices and update");
            }
            break;
        }
        case ScatterUpdateMode::ScatterNDUpdate: {
            if (indicesDim[indicesRank - 1] != Shape::UNDEFINED_DIM) {
                size_t k = indicesDim[indicesRank - 1];
                if (k > srcRank) {
                    OPENVINO_THROW(errorPrefix,
                                   "' do not have an correct indices' last dimension value, ",
                                   "which should be smaller than or equal to input tensor rank");
                }

                size_t tupleRank = indicesRank - 1;
                VectorDims expectUpdateShape(tupleRank + srcRank - k, 0);
                int updateAxisIter = 0;
                for (size_t ri = 0; ri < tupleRank; ri++) {
                    expectUpdateShape[updateAxisIter] = indicesDim[ri];
                    updateAxisIter++;
                }
                for (size_t rd = k; rd < srcRank; rd++) {
                    expectUpdateShape[updateAxisIter] = srcDataDim[rd];
                    updateAxisIter++;
                }
                if (expectUpdateShape.size() != updateRank) {
                    OPENVINO_THROW(errorPrefix,
                                   " do not have matched tensor rank relationship for input, indices and update");
                }
                for (size_t ru = 0; ru < updateRank; ru++) {
                    if (!dimsEqualWeak(updateDim[ru], expectUpdateShape[ru])) {
                        OPENVINO_THROW(errorPrefix,
                                       " do not have matched tensor shape relationship for input, indices and update");
                    }
                }
            }
            break;
        }
        case ScatterUpdateMode::ScatterElementsUpdate: {
            if (srcRank != indicesRank || srcRank != updateRank) {
                OPENVINO_THROW(errorPrefix, " do not have the same tensor rank for input, indices and update");
            }
            for (size_t ri = 0; ri < indicesRank; ri++) {
                if (!dimsEqualWeak(indicesDim[ri], updateDim[ri])) {
                    OPENVINO_THROW(errorPrefix, " do not have the same tensor shape for indices and update");
                }
            }
            break;
        }
        default: {
            OPENVINO_THROW(errorPrefix, " is not supported");
        }
    }

    indicesPrec = getOriginalInputPrecisionAtPort(INDICES_ID);
    auto indicesType = DnnlExtensionUtils::ElementTypeToDataType(indicesPrec);
    indicesSize = DnnlExtensionUtils::sizeOfDataType(indicesType);
    if (indicesSize >= 8) {
        indicesPrec = ov::element::i64;
        indicesSize = 8;
    } else {
        indicesPrec = ov::element::i32;
        indicesSize = 4;
    }

    if (axisRelaxed) {
        axisPrec = getOriginalInputPrecisionAtPort(AXIS_ID);
        auto axisType = DnnlExtensionUtils::ElementTypeToDataType(axisPrec);
        axisSize = DnnlExtensionUtils::sizeOfDataType(axisType);
        if (axisSize >= 8) {
            axisPrec = ov::element::i64;
            axisSize = 8;
        } else {
            axisPrec = ov::element::i32;
            axisSize = 4;
        }
    }

    dataPrec = getOriginalInputPrecisionAtPort(DATA_ID);
    dataSize = dataPrec.size();

    bool canBeInplace = !getParentEdgeAt(DATA_ID)->getParent()->isConstant();

    std::vector<PortConfigurator> inPortConfig{{LayoutType::ncsp, dataPrec, false, canBeInplace ? 0 : -1},
                                                {LayoutType::ncsp, indicesPrec},
                                                {LayoutType::ncsp, dataPrec}};
    if (axisRelaxed)
        inPortConfig.emplace_back(LayoutType::ncsp, axisPrec);
    addSupportedPrimDesc(inPortConfig,
                         {{LayoutType::ncsp, dataPrec, false, canBeInplace ? 0 : -1}},
                          impl_desc_type::unknown);
}

bool ScatterUpdate::needPrepareParams() const {
    return false;
}

void ScatterUpdate::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

int64_t ScatterUpdate::getIndicesValue(uint8_t *indices, size_t offset) {
    auto *indicesPtr = indices + offset * indicesSize;
    int64_t ret = 0;
    if (indicesSize == 4) {
        auto *indicesPtr32 = reinterpret_cast<int32_t*>(indicesPtr);
        ret = *indicesPtr32;
    } else {
        auto *indicesPtr64 = reinterpret_cast<int64_t*>(indicesPtr);
        ret = *indicesPtr64;
    }
    return ret;
}

// 5D example:
// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
static std::vector<size_t> getBlockND(const VectorDims& shape) {
    size_t shapeRank = shape.size();
    std::vector<size_t> blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i+1];
    }
    return blockND;
}

namespace scatter_elements_update {

class ReduceMultiply {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data *= *src_data;
    }
};

class ReduceAdd {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data += *src_data;
    }
};

class ReduceMean {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data += *src_data;
    }
};

class ReduceMaximum {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data = std::isnan(*src_data) ? *src_data : std::max(*dst_data, *src_data);
    }
};

class ReduceMinimum {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data = std::isnan(*src_data) ? *src_data : std::min(*dst_data, *src_data);
    }
};

class ReduceNone {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data = *src_data;
    }
};

template <typename T>
static T reduction_neutral_value(const ScatterUpdate::Reduction reduction_type) {
    switch (reduction_type) {
    case ScatterUpdate::Reduction::MAX:
        return std::numeric_limits<T>::lowest();
    case ScatterUpdate::Reduction::MIN:
        return std::numeric_limits<T>::max();
    case ScatterUpdate::Reduction::PROD:
        return T{1};
    case ScatterUpdate::Reduction::SUM:
    case ScatterUpdate::Reduction::MEAN:
    case ScatterUpdate::Reduction::NONE:
        return T{0};
    default:
        OPENVINO_THROW("Neutral value not available for this type of reduction");
        return 0;
    }
}

static ReduceMultiply reduce_multiply;
static ReduceAdd reduce_add;
static ReduceMean reduce_mean;
static ReduceMaximum reduce_maximum;
static ReduceMinimum reduce_minimum;
static ReduceNone data_assign;

static inline void getCoordinate(VectorDims& coordinate, size_t offset, const VectorDims& shape) {
    size_t shapeRank = shape.size();
    for (int i = shapeRank - 1; i >= 0; i--) {
        coordinate[i] = offset % shape[i];
        offset /= shape[i];
    }
}

struct TensorIterator {
    TensorIterator(const VectorDims& squashed_shape, const int64_t squashed_axis) : m_squashed_shape(squashed_shape), m_squashed_axis(squashed_axis) {
        OPENVINO_ASSERT(m_squashed_shape[m_squashed_axis] == 1);
    }

    std::array<size_t, 2> startover(const size_t start, const std::vector<size_t>& dataBlockND, const std::vector<size_t>& indicesBlockND) {
        m_tensorIter.resize(m_squashed_shape.size(), 0);
        getCoordinate(m_tensorIter, start, m_squashed_shape);

        size_t i, dst_idx = 0, indices_idx = 0;
        for (i = 0; i < static_cast<size_t>(m_squashed_axis); ++i) {
            dst_idx += m_tensorIter[i] * dataBlockND[i + 1];
            indices_idx += m_tensorIter[i] * indicesBlockND[i + 1];
        }
        for (i++; i < m_squashed_shape.size(); ++i) {
            dst_idx += m_tensorIter[i] * dataBlockND[i + 1];
            indices_idx += m_tensorIter[i] * indicesBlockND[i + 1];
        }

        return {dst_idx, indices_idx};
    }

    void increment(std::array<size_t, 2>& offsets, const std::vector<size_t>& dataBlockND, const std::vector<size_t>& indicesBlockND) {
        for (int64_t j = m_squashed_shape.size() - 1; j >= 0; j--) {
            m_tensorIter[j]++;
            if (m_tensorIter[j] < m_squashed_shape[j]) { // no need check if (j != axis) as it is squashed
                offsets[0] += dataBlockND[j + 1];
                offsets[1] += indicesBlockND[j + 1];
                break;
            } else {
                m_tensorIter[j] = 0;
                size_t i = 0;
                for (offsets[0] = 0, offsets[1] = 0; i < m_squashed_axis; ++i) {
                    offsets[0] += m_tensorIter[i] * dataBlockND[i + 1];
                    offsets[1] += m_tensorIter[i] * indicesBlockND[i + 1];
                }
                for (i++; i < m_squashed_shape.size(); ++i) {
                    offsets[0] += m_tensorIter[i] * dataBlockND[i + 1];
                    offsets[1] += m_tensorIter[i] * indicesBlockND[i + 1];
                }
            }
        }
    }

    VectorDims m_tensorIter;
    const VectorDims m_squashed_shape;
    const size_t m_squashed_axis;
};

// output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
// output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
// output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2.
template <typename DataType, typename IndexType, typename func_t>
void scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates,
                            int axis, const bool use_init_val, const ScatterUpdate::Reduction reduction_type, func_t& kernel_func) {
    DataType *dataPtr = mem_data->getDataAs<DataType>();
    DataType *updatePtr = mem_updates->getDataAs<DataType>();
    IndexType *indicesPtr = mem_indices->getDataAs<IndexType>();

    const auto& data_shape = mem_data->getStaticDims();
    const auto& indices_shape = mem_indices->getStaticDims();
    size_t updates_rank = indices_shape.size();

    const int64_t data_dim_size = static_cast<int64_t>(data_shape[axis]);
    const auto index_dim_size = indices_shape[axis];

    if (axis < 0)
        axis += updates_rank;

    VectorDims squashed_indices_shape(indices_shape);
    squashed_indices_shape[axis] = 1;

    const std::vector<size_t> dataBlockND = getBlockND(data_shape);
    const std::vector<size_t> indicesBlockND = getBlockND(indices_shape);

    // process serially along 'axis' dimension because of data dependency brought by duplicated value in indices
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);
        TensorIterator tensorItr(squashed_indices_shape, axis);

        // When *use_init_val* attribute is false, we need to substitute the copied values at target locations with values that
        // will not affect the particular reduction algorithms.
        if (!use_init_val) {
            const auto value = reduction_neutral_value<DataType>(reduction_type);
            auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
            for (size_t worker = start; worker < end; worker++) {
                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    IndexType idxValue = *(indicesPtr + indices_offset);
                    if (idxValue < 0) idxValue += data_dim_size;
                    ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                    auto dst = dataPtr + (offsets[0] + idxValue * dataBlockND[axis + 1]);
                    *dst = value;
                    indices_offset += indicesBlockND[axis + 1];
                }

                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        }

        // Apply the Reduce function in an element-wise fashion. For better performance,
        // when axis is the last dimension, we iterate along axis in the inner loop; otherwise we iterate axis
        // in the outer loop.
        auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
        if (axis == static_cast<int>(updates_rank - 1)) {
            for (size_t worker = start; worker < end; worker++) {
                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    IndexType idxValue = *(indicesPtr + indices_offset);
                    if (idxValue < 0) idxValue += data_dim_size;
                    ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                    auto dst = dataPtr + (offsets[0] + idxValue * dataBlockND[axis + 1]);
                    auto src = updatePtr + indices_offset;
                    kernel_func(dst, src);
                    indices_offset += indicesBlockND[axis + 1];
                }
                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        } else {
            // For better performance, the offsets of dst and indices are cached in the first iteration of outer loop, and reused
            // in the remaining iterations.
            std::vector<size_t> dst_offsets(end-start+1, offsets[0]);  // one extra to avoid overflow at the last iteration of inner loop
            std::vector<size_t> indices_offsets(end-start+1, offsets[1]);
            size_t *ptr_dst_offset = &dst_offsets[0];
            size_t *ptr_indices_offset = &indices_offsets[0];
            for (size_t worker = start; worker < end; worker++) { // idx = 0
                IndexType idxValue = *(indicesPtr + *ptr_indices_offset);
                if (idxValue < 0) idxValue += data_dim_size;
                ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                auto dst = dataPtr + (*ptr_dst_offset + idxValue * dataBlockND[axis + 1]);
                auto src = updatePtr + *ptr_indices_offset;
                kernel_func(dst, src);

                // increment once for all
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
                *++ptr_dst_offset = offsets[0];
                *++ptr_indices_offset = offsets[1];
            }
            for (size_t idx = 1; idx < index_dim_size; idx++) {
                ptr_indices_offset = &indices_offsets[0];
                ptr_dst_offset = &dst_offsets[0];
                for (size_t worker = start; worker < end; worker++) {
                    auto indices_offset = *ptr_indices_offset + idx * indicesBlockND[axis + 1];
                    IndexType idxValue = *(indicesPtr + indices_offset);
                    if (idxValue < 0) idxValue += data_dim_size;
                    ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                    auto dst = dataPtr + (*ptr_dst_offset + idxValue * dataBlockND[axis + 1]);
                    auto src = updatePtr + indices_offset;
                    kernel_func(dst, src);
                    ptr_indices_offset++;
                    ptr_dst_offset++;
                }
            }
        }
    });
}

// We specialize ReduceMean to avoid spoil performance of other reduce methods, as otherwise condition branch
// were used in loops.
template <typename DataType, typename IndexType>
void scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates,
                            int axis, const bool use_init_val, const ScatterUpdate::Reduction reduction_type, ReduceMean& kernel_func) {
    OPENVINO_ASSERT(reduction_type == ScatterUpdate::Reduction::MEAN, "The reduction type should be MEAN here.");
    DataType *dataPtr = mem_data->getDataAs<DataType>();
    DataType *updatePtr = mem_updates->getDataAs<DataType>();
    IndexType *indicesPtr = mem_indices->getDataAs<IndexType>();

    const auto& data_shape = mem_data->getStaticDims();
    const auto& indices_shape = mem_indices->getStaticDims();
    size_t updates_rank = indices_shape.size();

    const int64_t data_dim_size = static_cast<int64_t>(data_shape[axis]);
    const auto index_dim_size = indices_shape[axis];

    if (axis < 0)
        axis += updates_rank;

    VectorDims squashed_indices_shape(indices_shape);
    squashed_indices_shape[axis] = 1;

    const std::vector<size_t> dataBlockND = getBlockND(data_shape);
    const std::vector<size_t> indicesBlockND = getBlockND(indices_shape);

    // process serially along 'axis' dimension because of data dependency brought by duplicated value in indices
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);
        TensorIterator tensorItr(squashed_indices_shape, axis);

        // When *use_init_val* attribute is false, we need to substitute the copied values at target locations with values that
        // will not affect the particular reduction algorithms.
        if (!use_init_val) {
            const auto value = reduction_neutral_value<DataType>(reduction_type);
            auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
            for (size_t worker = start; worker < end; worker++) {
                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    IndexType idxValue = *(indicesPtr + indices_offset);
                    if (idxValue < 0) idxValue += data_dim_size;
                    ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                    auto dst = dataPtr + (offsets[0] + idxValue * dataBlockND[axis + 1]);
                    *dst = value;
                    indices_offset += indicesBlockND[axis + 1];
                }

                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        }

        // Apply the Reduce function in an element-wise fashion. For better performance,
        // when axis is the last dimension, we iterate along axis in the inner loop; otherwise we iterate axis
        // in the outer loop.
        auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
        if (axis == static_cast<int>(updates_rank - 1)) {
            for (size_t worker = start; worker < end; worker++) {
                std::unordered_map<size_t, int64_t> mean_reduction_counters;  // (idxValue, num_sums) for current worker

                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    IndexType idxValue = *(indicesPtr + indices_offset);
                    if (idxValue < 0) idxValue += data_dim_size;
                    ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                    auto dst = dataPtr + (offsets[0] + idxValue * dataBlockND[axis + 1]);
                    auto src = updatePtr + indices_offset;
                    kernel_func(dst, src);
                    indices_offset += indicesBlockND[axis + 1];

                    mean_reduction_counters[idxValue] += 1;
                }

                // average
                for (const auto& counter : mean_reduction_counters) {
                    auto dst = dataPtr + (offsets[0] + counter.first * dataBlockND[axis + 1]);
                    const auto N = counter.second + static_cast<int32_t>(use_init_val);
                    *dst = static_cast<DataType>(static_cast<double>(*dst) / N);
                }

                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        } else {
            // For better performance, the offsets of dst and indices are cached in the first iteration of outer loop, and reused
            // in the remaining iterations.
            std::unordered_map<DataType*, int64_t> mean_reduction_counters;     // (dst_addr, num_sums) for all workers

            std::vector<size_t> dst_offsets(end-start+1, offsets[0]);  // one extra to avoid overflow at the last iteration of inner loop
            std::vector<size_t> indices_offsets(end-start+1, offsets[1]);
            size_t *ptr_dst_offset = &dst_offsets[0];
            size_t *ptr_indices_offset = &indices_offsets[0];
            for (size_t worker = start; worker < end; worker++) { // idx = 0
                IndexType idxValue = *(indicesPtr + *ptr_indices_offset);
                if (idxValue < 0) idxValue += data_dim_size;
                ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                auto dst = dataPtr + (*ptr_dst_offset + idxValue * dataBlockND[axis + 1]);
                auto src = updatePtr + *ptr_indices_offset;
                kernel_func(dst, src);

                mean_reduction_counters[dst] += 1;

                // increment once for all
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
                *++ptr_dst_offset = offsets[0];
                *++ptr_indices_offset = offsets[1];
            }
            for (size_t idx = 1; idx < index_dim_size; idx++) {
                ptr_indices_offset = &indices_offsets[0];
                ptr_dst_offset = &dst_offsets[0];
                for (size_t worker = start; worker < end; worker++) {
                    auto indices_offset = *ptr_indices_offset + idx * indicesBlockND[axis + 1];
                    IndexType idxValue = *(indicesPtr + indices_offset);
                    if (idxValue < 0) idxValue += data_dim_size;
                    ASSERT_DEBUG_ONLY(idxValue < data_dim_size && idxValue >= 0, "invalid index value.");
                    auto dst = dataPtr + (*ptr_dst_offset + idxValue * dataBlockND[axis + 1]);
                    auto src = updatePtr + indices_offset;
                    kernel_func(dst, src);
                    mean_reduction_counters[dst] += 1;
                    ptr_indices_offset++;
                    ptr_dst_offset++;
                }
            }

            // average
            for (const auto& counter : mean_reduction_counters) {
                auto dst = counter.first;
                const auto N = counter.second + static_cast<int32_t>(use_init_val);
                *dst = static_cast<DataType>(static_cast<double>(*dst) / N);
            }
        }
    });
}

template <typename DT, typename IT>
void scatterElementsUpdate_target(const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr,
                        int axis, const bool use_init_val, const ScatterUpdate::Reduction reduction_type) {
    switch (reduction_type) {
    case ScatterUpdate::Reduction::NONE :
        scatterElementsUpdate<DT, IT>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type, data_assign);
        break;
    case ScatterUpdate::Reduction::SUM :
        scatterElementsUpdate<DT, IT>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type, reduce_add);
        break;
    case ScatterUpdate::Reduction::MAX :
        scatterElementsUpdate<DT, IT>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type, reduce_maximum);
        break;
    case ScatterUpdate::Reduction::MIN :
        scatterElementsUpdate<DT, IT>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type, reduce_minimum);
        break;
    case ScatterUpdate::Reduction::PROD:
        scatterElementsUpdate<DT, IT>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type, reduce_multiply);
        break;
    case ScatterUpdate::Reduction::MEAN :
        scatterElementsUpdate<DT, IT>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type, reduce_mean);
        break;
    default :
        OPENVINO_THROW("unsupported reduce");
        break;
    }
}

template<typename DT>
void dispatch_IT(const ov::element::Type& indices_precision,
                    const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr,
                    int axis, const bool use_init_val, const ScatterUpdate::Reduction reduction_type) {
    switch (indices_precision) {
        case ov::element::i32:
            scatterElementsUpdate_target<DT, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        case ov::element::i64:
            scatterElementsUpdate_target<DT, int64_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        default:
            OPENVINO_THROW("Unsupported precision for indices ", indices_precision);
            break;
    }
}

static inline
void precision_dispatch(const ov::element::Type& data_precision, const ov::element::Type& indices_precision,
                        const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr,
                        int axis, const bool use_init_val, const ScatterUpdate::Reduction reduction_type) {
    switch (data_precision) {
        case ov::element::f32:
            dispatch_IT<float>(indices_precision, dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        case ov::element::bf16:
            dispatch_IT<ov::bfloat16>(indices_precision, dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        case ov::element::f16:
            dispatch_IT<ov::float16>(indices_precision, dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        case ov::element::i32:
            dispatch_IT<int32_t>(indices_precision, dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        default:
            OPENVINO_THROW("Unsupported precision for data and updates ", data_precision);
            break;
    }
}

};  // namespace scatter_elements_update

void ScatterUpdate::scatterElementsUpdate(const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr,
                        int axis, const bool use_init_val, const ScatterUpdate::Reduction reduction_type) {
    scatter_elements_update::precision_dispatch(dataPrec, indicesPrec,
                                                dstMemPtr, indicesMemPtr, updateMemPtr,
                                                axis, use_init_val, reduction_type);
}

void ScatterUpdate::execute(dnnl::stream strm) {
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto indicesMemPtr = getSrcMemoryAtPort(INDICES_ID);
    auto updateMemPtr = getSrcMemoryAtPort(UPDATE_ID);

    uint8_t *dstPtr = dstMemPtr->getDataAs<uint8_t>();
    uint8_t *srcPtr = srcMemPtr->getDataAs<uint8_t>();
    uint8_t *indicesPtr = indicesMemPtr->getDataAs<uint8_t>();
    uint8_t *updatePtr = updateMemPtr->getDataAs<uint8_t>();

    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    size_t srcRank = srcDataDim.size();

    // 1d short vector scatter update optimized for shape inference subgraph
    if (scatterUpdateMode == ScatterUpdateMode::ScatterUpdate && srcDataDim.size() == 1 && indicesDim.size() <= 1 &&
        indicesPrec == ov::element::i32 && dataPrec == ov::element::i32 && srcDataDim[0] <= 64) {
        auto updateDims = updateMemPtr->getStaticDims();
        if (updateDims.size() <= 1) {
            DEBUG_LOG(getName(), " exec1DCase");
            auto updateCnt = (updateDims.size() == 0) ? 1 : updateDims[0];
            auto srcLength = srcMemPtr->getStaticDims()[0];
            auto* psrc = reinterpret_cast<int32_t*>(srcPtr);
            auto* pdst = reinterpret_cast<int32_t*>(dstPtr);
            for (size_t i = 0; i < srcLength; i++) {
                pdst[i] = psrc[i];
            }
            auto* pindices = reinterpret_cast<int32_t*>(indicesPtr);
            auto* pupdate = reinterpret_cast<int32_t*>(updatePtr);
            for (size_t i = 0; i < updateCnt; i++) {
                pdst[pindices[i]] = pupdate[i];
            }
            return;
        }
    }

    int axis = 0;
    if (axisRelaxed) {
        auto axisMemPtr = getSrcMemoryAtPort(AXIS_ID);
        uint8_t *axisPtr = axisMemPtr->getDataAs<uint8_t>();
        if (axisSize == 4) {
            auto *axisPtr32 = reinterpret_cast<int32_t*>(axisPtr);
            axis = *axisPtr32;
        } else {
            auto *axisPtr64 = reinterpret_cast<int64_t*>(axisPtr);
            axis = *axisPtr64;
        }

        if (axis >= static_cast<int>(srcRank) || axis < (static_cast<int>(srcRank) * - 1)) {
            OPENVINO_THROW(errorPrefix
           , " should have axis value in range [-r, r - 1], where r is the rank of input data");
        }
        axis = axis < 0 ? (axis + srcRank) : axis;

        size_t srcDimAxis = srcDataDim[axis];
        std::vector<size_t> indicesBlockND = getBlockND(indicesDim);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(indicesBlockND[0], nthr, ithr, start, end);
            for (size_t i = start; i < end; i++) {
                int64_t idxValue =  getIndicesValue(indicesPtr, i);
                if (idxValue >= static_cast<int64_t>(srcDimAxis) ||
                    (idxValue < 0 && scatterUpdateMode != ScatterUpdateMode::ScatterElementsUpdate)) {
                    OPENVINO_THROW(errorPrefix
                              , " have indices value that points to non-existing output tensor element");
                }
            }
        });

        if (scatterUpdateMode == ScatterUpdateMode::ScatterUpdate) {
            VectorDims indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
            VectorDims updateDim = getParentEdgeAt(UPDATE_ID)->getMemory().getStaticDims();
            size_t indicesRank = indicesDim.size();
            size_t updateRank = updateDim.size();
            VectorDims expectUpdateShape(srcRank + indicesRank - 1, 0);
            int axisIter = 0;
            for (size_t rs = 0; rs < srcRank; rs++) {
                if (rs != static_cast<size_t>(axis)) {
                    expectUpdateShape[axisIter] = srcDataDim[rs];
                    axisIter++;
                } else {
                    for (size_t ri = 0; ri < indicesRank; ri++) {
                        expectUpdateShape[axisIter] = indicesDim[ri];
                        axisIter++;
                    }
                }
            }
            if (updateRank > expectUpdateShape.size())
                OPENVINO_THROW(errorPrefix,
                               " cannot update shape. New rank: ",
                               updateRank,
                               ", expected: ",
                               expectUpdateShape.size());
            for (size_t ru = 0; ru < updateRank; ru++) {
                if (updateDim[ru] != expectUpdateShape[ru]) {
                    OPENVINO_THROW(errorPrefix,
                                   " do not have matched tensor shape relationship for input, indices and update");
                }
            }
        }
    }

    if (srcPtr != dstPtr) {
        std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(srcBlockND[0], nthr, ithr, start, end);
            size_t size = (end - start) * dataSize;
            start *= dataSize;
            cpu_memcpy(dstPtr + start, srcPtr + start, size);
        });
    }

    if (isInputTensorAtPortEmpty(INDICES_ID)) {
        return;
    }

    switch (scatterUpdateMode) {
        case ScatterUpdateMode::ScatterUpdate: {
            scatterUpdate(indicesPtr, updatePtr, axis, dstPtr);
            break;
        }
        case ScatterUpdateMode::ScatterNDUpdate: {
            scatterNDUpdate(indicesPtr, updatePtr, dstPtr);
            break;
        }
        case ScatterUpdateMode::ScatterElementsUpdate: {
            scatterElementsUpdate(dstMemPtr, indicesMemPtr, updateMemPtr, axis, use_init_val, reduction_type);
            break;
        }
        default: {
            OPENVINO_THROW(errorPrefix, " is not supported");
        }
    }
}

// For the data tensor of shape [d_0, d_1, ..., d_n],
// and indices tensor of shape [i_0, i_1, ..., i_k].
// Updates tensor shape should be [d_0, d_1, ... d_(axis - 1), i_0, i_1, ..., i_k, d_(axis + 1), ..., d_n].
void ScatterUpdate::scatterUpdate(uint8_t *indices, uint8_t *update, int axis, uint8_t *dstData) {
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    const auto& updateDim = getParentEdgeAt(UPDATE_ID)->getMemory().getStaticDims();
    size_t indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
    std::vector<size_t> updateBlockND = getBlockND(updateDim);

    const size_t mulIdentity = 1;
    size_t idxLength = mulIdentity;
    for (size_t ri = 0; ri < indicesRank; ri++) {
        idxLength *= indicesDim[ri];
    }
    size_t batchToUpdate = mulIdentity;
    for (int x = 0; x < axis; x++) {
        batchToUpdate *= srcDataDim[x];
    }
    // blockToUpdate is srcBlockND[axis + 1], also is updateBlockND[axis + indicesRank]
    size_t blockToUpdate = srcBlockND[axis + 1];
    size_t blockToUpdateSize = blockToUpdate * dataSize;

    parallel_for2d(batchToUpdate, idxLength, [&](size_t b, size_t idx) {
        int64_t idxValue = getIndicesValue(indices, idx);
        uint8_t *dstEntry = dstData + (b * srcBlockND[axis] + idxValue * blockToUpdate) * dataSize;
        uint8_t *updateEntry = update + (b * updateBlockND[axis] + idx * blockToUpdate) * dataSize;
        cpu_memcpy(dstEntry, updateEntry, blockToUpdateSize);
    });
}

// indices is a (q-1)-dimension tensor of k-tuple,
// k is indices.shape[-1] and should not be greater than rank of input, q is rank of indicies.
// updates is a (q-1)-dimension tensor of replacement-slice-values
void ScatterUpdate::scatterNDUpdate(uint8_t *indices, uint8_t *update, uint8_t *dstData) {
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    size_t indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);

    size_t k = indicesDim[indicesRank - 1];
    size_t idxTupleNum = 1;
    for (size_t ri = 0; ri < indicesRank - 1; ri++) {
        idxTupleNum *= indicesDim[ri];
    }

    size_t sizeToUpdate = srcBlockND[k] * dataSize;
    parallel_for(idxTupleNum, [&](size_t tupleIdx) {
        size_t indicesOffset = tupleIdx * k;
        size_t dstOffset = 0;
        for (size_t i = 0; i < k; i++) {
            int64_t idxValue = getIndicesValue(indices, indicesOffset + i);
            if (idxValue < 0) {
                // Negative value for indices means counting backwards from the end.
                idxValue += srcDataDim[i];
            }
            dstOffset += idxValue * srcBlockND[i + 1];
        }
        dstOffset *= dataSize;
        size_t updateOffset = tupleIdx * sizeToUpdate;
        cpu_memcpy(dstData + dstOffset, update + updateOffset, sizeToUpdate);
    });
}


bool ScatterUpdate::created() const {
    return getType() == Type::ScatterUpdate
            || getType() == Type::ScatterElementsUpdate
            || getType() == Type::ScatterNDUpdate;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
