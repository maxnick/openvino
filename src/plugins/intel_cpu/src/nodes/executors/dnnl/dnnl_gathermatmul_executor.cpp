// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_gathermatmul_executor.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "nodes/common/blocked_desc_creator.h"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {
class OffsetHelper {
public:
    static OffsetHelper createOffsetHelper(const MemoryPtr& mem) {
        static const VectorDims empty_dims;
        std::bitset<2> broadcast_mask;
        if (nullptr == mem || mem->getDesc().empty()) {
            return {nullptr, empty_dims, broadcast_mask, 0};
        }
        return createOffsetHelper(*mem);
    }

    static OffsetHelper createOffsetHelper(const IMemory& mem) {
        std::bitset<2> broadcast_mask;
        auto* base_ptr = static_cast<uint8_t*>(mem.getData());
        auto desc = mem.getDescWithType<BlockedMemoryDesc>();
        const auto& strides = desc->getStrides();
        const auto prc = desc->getPrecision();
        const auto& shape = desc->getShape().getStaticDims();
        for (size_t i = 0; i < shape.size() && i < 2; i++) {
            if (shape[i] == 1) {
                broadcast_mask.set(i);
            }
        }
        return {base_ptr, strides, broadcast_mask, prc.bitwidth()};
    }

    void* operator()(size_t i0) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);
        return m_base_ptr + offset;
    }

    void* operator()(size_t i0, size_t i1) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        if (m_broadcast_mask.test(1)) {
            i1 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits + i1 * m_strides[1] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);
        return m_base_ptr + offset;
    }

    [[nodiscard]] void* get_base() const {
        return m_base_ptr;
    }

private:
    OffsetHelper(uint8_t* base_ptr, const VectorDims& strides, std::bitset<2> broadcast_mask, size_t num_bits)
        : m_base_ptr(base_ptr),
          m_strides(strides),
          m_num_bits(num_bits),
          m_broadcast_mask(broadcast_mask) {}

    uint8_t* m_base_ptr = nullptr;
    const VectorDims& m_strides;
    size_t m_num_bits;
    std::bitset<2> m_broadcast_mask;
};

dnnl::memory::desc bias1DDesc(const MemoryPtr& biasMem) {
    const auto N = static_cast<dnnl::memory::dim>(biasMem->getStaticDims().back());
    const auto dt = DnnlExtensionUtils::ElementTypeToDataType(biasMem->getDesc().getPrecision());
    return dnnl::memory::desc({N}, dt, dnnl::memory::format_tag::a);
}

// Returns the ncsp dnnl descriptor for one gather-axis slice (strips the leading G dim).
dnnl::memory::desc gatherSliceDesc(const MemoryPtr& mem) {
    const auto& fullDims = mem->getStaticDims();
    const auto dt = DnnlExtensionUtils::ElementTypeToDataType(mem->getDesc().getPrecision());
    const dnnl::memory::dims sliceDims(fullDims.begin() + 1, fullDims.end());
    return dnnl::memory::desc(sliceDims, dt, dnnl::memory::format_tag::ab);
}

Dim normalizeM(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);
    } else {
        M = rnd_up(M, 256);
    }
    return M;
}

template <typename DescFn>
std::optional<dnnl::memory> toSliceMemory(const MemoryPtr& mem, DescFn descFn, const dnnl::engine& eng) {
    if (!mem || mem->getDesc().empty()) {
        return std::nullopt;
    }
    return dnnl::memory(descFn(mem), eng, mem->getData());
}

MemoryArgs makeSliceMemoryArgs(const dnnl::memory::desc& src_md,
                               const dnnl::memory::desc& wei_md,
                               const dnnl::memory::desc& dst_md,
                               const std::optional<dnnl::memory>& bias,
                               const std::optional<dnnl::memory>& scales,
                               const std::optional<dnnl::memory>& zp,
                               const dnnl::engine& eng) {
    auto wrap = [&eng](const dnnl::memory& m) {
        return std::make_shared<Memory>(eng, DnnlExtensionUtils::makeDescriptor(m.get_desc()), m.get_data_handle());
    };

    MemoryArgs args;
    args[ARG_SRC] = std::make_shared<Memory>(eng, DnnlExtensionUtils::makeDescriptor(src_md));
    args[ARG_WEI] = std::make_shared<Memory>(eng, DnnlExtensionUtils::makeDescriptor(wei_md));
    args[ARG_DST] = std::make_shared<Memory>(eng, DnnlExtensionUtils::makeDescriptor(dst_md));
    args[ARG_BIAS] = bias ? wrap(*bias) : std::make_shared<Memory>(eng, MemoryDescUtils::makeEmptyDesc());

    if (scales) {
        args[ARG_WEI | ARG_ATTR_SCALES] = wrap(*scales);
    }
    if (zp) {
        args[ARG_WEI | ARG_ATTR_ZERO_POINTS] = wrap(*zp);
    }
    return args;
}

dnnl_primitive_args makePrimArgs(const DnnlFCPrimitivePtr& prim,
                                 const DnnlShapeAgnosticDataPtr& shapeAgnosticData,
                                 const MemoryPtr& biasMem,
                                 const MemoryPtr& scratchpadMem,
                                 const dnnl::engine& eng) {
    dnnl_primitive_args args;
    args[DNNL_ARG_SRC] = dnnl::memory(prim->srcDesc()->getDnnlDesc(), eng, DNNL_MEMORY_NONE);
    args[DNNL_ARG_DST] = dnnl::memory(prim->dstDesc()->getDnnlDesc(), eng, DNNL_MEMORY_NONE);
    args[DNNL_ARG_WEIGHTS] = dnnl::memory(prim->weightsDesc()->getDnnlDesc(), eng, DNNL_MEMORY_NONE);

    if (biasMem && !biasMem->getDesc().empty()) {
        args[DNNL_ARG_BIAS] = dnnl::memory(bias1DDesc(biasMem), eng, DNNL_MEMORY_NONE);
    }

    const auto& dnnlArgs = shapeAgnosticData->m_primAttrs.dnnlArgs;
    const auto& cpuArgs = shapeAgnosticData->m_primAttrs.cpuArgs;
    for (const int key : {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS}) {
        if (dnnlArgs.count(key) && cpuArgs.count(key)) {
            const auto& dims = cpuArgs.at(key)->getStaticDims();
            auto dnnlDims = DnnlExtensionUtils::convertToDnnlDims(dims);
            auto dt = dnnlArgs.at(key).get_desc().get_data_type();
            args[key] =
                dnnl::memory(dnnl::memory::desc(dnnlDims, dt, dnnl::memory::format_tag::ba), eng, DNNL_MEMORY_NONE);
        }
    }

    args[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
    return args;
}
}  // namespace

bool GatherMatmulDnnlExecutor::supports([[maybe_unused]] const GatherMatmulConfig& config) {
#ifdef OPENVINO_ARCH_X86_64
    if ((config.descs.count(ARG_SRC) != 0U) && !config.descs.at(ARG_SRC)->empty()) {
        const auto src_prc = config.descs.at(ARG_SRC)->getPrecision();
        if (!any_of(src_prc, ov::element::f32, ov::element::bf16)) {
            return false;
        }
    }
    if ((config.descs.count(ARG_WEI) != 0U) && !config.descs.at(ARG_WEI)->empty()) {
        const auto wei_prc = config.descs.at(ARG_WEI)->getPrecision();
        if (any_of(wei_prc, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4)) {
            if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
                return false;
            }
        }
    }
    return true;
#else
    return false;
#endif
}

GatherMatmulDnnlExecutor::GatherMatmulDnnlExecutor([[maybe_unused]] const GatherMatmulAttrs& attrs,
                                                   const MemoryArgs& memory,
                                                   const ExecutorContext::CPtr& context)
    : m_context(context) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& srcMemory = memory.at(ARG_SRC);

    auto src_precision = srcMemory->getDesc().getPrecision();
    auto weights_precision = weightsMemory->getDesc().getPrecision();

#ifdef OPENVINO_ARCH_X86_64
    m_bf16AmxMode =
        (src_precision == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx));
#endif

    const auto& weiDims = weightsMemory->getShape().getStaticDims();
    const dnnl::memory::dim N = weiDims[weiDims.size() - 2];
    const dnnl::memory::dim K = weiDims[weiDims.size() - 1];

    const auto& scalesMem = memory.at(ARG_SRC_3);
    const auto& zpMem = memory.at(ARG_SRC_4);
    const auto& biasMem = memory.at(ARG_BIAS);

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc dst_md({1, N},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::ab);

    const FCAttrs fcAttrs{};
    const auto& eng = context->getEngine();
    auto sliceArgs = makeSliceMemoryArgs(src_md,
                                         weights_md,
                                         dst_md,
                                         toSliceMemory(biasMem, bias1DDesc, eng),
                                         toSliceMemory(scalesMem, gatherSliceDesc, eng),
                                         toSliceMemory(zpMem, gatherSliceDesc, eng),
                                         eng);
    auto shapeAgnosticData = DnnlFCPrimitive::createShapeAgnosticData(fcAttrs, sliceArgs, context, false);

    m_gemvPrim = DnnlFCPrimitive::create(sliceArgs, fcAttrs, context, shapeAgnosticData);
    m_implType = m_gemvPrim->implType();

    auto gemvWeightsDesc = MemoryDescUtils::convertToBlockedMemoryDesc(m_gemvPrim->weightsDesc());

    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
        VectorDims newDims = {batchDim};
        newDims.insert(newDims.end(), weightsDims.begin(), weightsDims.end());
        VectorDims newBlockDims = {batchDim};
        newBlockDims.insert(newBlockDims.end(), weightsBlockDims.begin(), weightsBlockDims.end());
        VectorDims newOrder(weightsOrder.size() + 1);
        newOrder[0] = 0;
        for (size_t i = 0; i < weightsOrder.size(); i++) {
            newOrder[i + 1] = weightsOrder[i] + 1;
        }
        return MemoryDescUtils::convertToDnnlMemoryDesc(
            std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(newDims), newBlockDims, newOrder));
    };

    auto targetWeightsDesc = addBatchDim(gemvWeightsDesc, weiDims[0]);
    auto srcWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightsMemory->getDescPtr());

    m_weightsMemory = utils::prepareWeightsMemory(srcWeightsDesc,
                                                  targetWeightsDesc,
                                                  weightsMemory,
                                                  eng,
                                                  context->getRuntimeCache(),
                                                  context->getWeightsCache(),
                                                  context->getPrivateWeightCache(),
                                                  context->getThreadPool());

    const auto& primCpuArgs = shapeAgnosticData->m_primAttrs.cpuArgs;
    auto repackBatched = [&eng, &addBatchDim](const MemoryPtr& srcMem, const MemoryPtr& postPrepackSlice) -> MemoryPtr {
        const size_t G = srcMem->getShape().getStaticDims()[0];
        auto dstSliceBlockedDesc = MemoryDescUtils::convertToBlockedMemoryDesc(
            MemoryDescUtils::convertToDnnlMemoryDesc(postPrepackSlice->getDescPtr()));
        auto result = std::make_shared<Memory>(eng, addBatchDim(dstSliceBlockedDesc, G));
        result->load(*srcMem, false, false);
        return result;
    };

    if (scalesMem && !scalesMem->getDesc().empty() && primCpuArgs.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)) {
        auto postPrepackScale = primCpuArgs.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
        m_scalesMemory = repackBatched(std::const_pointer_cast<IMemory>(scalesMem), postPrepackScale);
    }

    if (zpMem && !zpMem->getDesc().empty() && primCpuArgs.count(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS)) {
        auto postPrepackZp = primCpuArgs.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
        m_zpMemory = repackBatched(std::const_pointer_cast<IMemory>(zpMem), postPrepackZp);
    }

    m_gemvScratchpad = context->getScratchPad()->createScratchPadMem(m_gemvPrim->scratchPadDesc());

    m_gemvArgs = makePrimArgs(m_gemvPrim, shapeAgnosticData, biasMem, m_gemvScratchpad, eng);
}

bool GatherMatmulDnnlExecutor::update(const MemoryArgs& memory) {
    if (!m_bf16AmxMode) {
        return true;
    }

    const auto& srcMem = memory.at(ARG_SRC);
    const auto& srcShape = srcMem->getStaticDims();
    if (Dim{1} == srcShape[1]) {
        return true;
    }

    const Dim M = normalizeM(srcShape[1]);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const auto srcPrc = srcMem->getDesc().getPrecision();
    const auto& dstShape = memory.at(ARG_DST)->getStaticDims();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    const dnnl::memory::dim N = static_cast<dnnl::memory::dim>(m_gemvPrim->weightsDesc()->getDnnlDesc().get_dims()[0]);
    const auto& eng = m_context->getEngine();

    dnnl::memory::desc src_md({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(srcShape[2])},
                              DnnlExtensionUtils::ElementTypeToDataType(srcPrc),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc dst_md({static_cast<dnnl::memory::dim>(M), N},
                              DnnlExtensionUtils::ElementTypeToDataType(srcPrc),
                              dnnl::memory::format_tag::ab);
    const auto& gemvWeiDnnlDesc = m_gemvPrim->weightsDesc()->getDnnlDesc();
    const dnnl::memory::desc weights_md(gemvWeiDnnlDesc.get_dims(),
                                        gemvWeiDnnlDesc.get_data_type(),
                                        dnnl::memory::format_tag::ab);

    const FCAttrs fcAttrs{};
    auto sliceArgs = makeSliceMemoryArgs(src_md,
                                         weights_md,
                                         dst_md,
                                         toSliceMemory(memory.at(ARG_BIAS), bias1DDesc, eng),
                                         toSliceMemory(m_scalesMemory, gatherSliceDesc, eng),
                                         toSliceMemory(m_zpMemory, gatherSliceDesc, eng),
                                         eng);
    auto shapeAgnosticData = DnnlFCPrimitive::createShapeAgnosticData(fcAttrs, sliceArgs, m_context, false);
    m_gemmPrim = DnnlFCPrimitive::create(sliceArgs, fcAttrs, m_context, shapeAgnosticData);

    const size_t srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);
    const size_t outputSize = rnd_up(m_tmpOutputDesc->getCurrentMemSize(), 64);
    const size_t gemmScratchSize = rnd_up(m_gemmPrim->scratchPadDesc()->getCurrentMemSize(), 64);
    const size_t totalSize = srcSize + outputSize + gemmScratchSize;

    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(scratchPadDesc);

    m_gemmScratchpad = std::make_shared<Memory>(eng,
                                                m_gemmPrim->scratchPadDesc(),
                                                m_tmpInpBuffer->getDataAs<uint8_t>() + srcSize + outputSize);

    m_gemmArgs = makePrimArgs(m_gemmPrim, shapeAgnosticData, memory.at(ARG_BIAS), m_gemmScratchpad, eng);

    return true;
}

void GatherMatmulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& cpu_parallel = m_context->getCpuParallel();
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& biasMem = memory.at(ARG_BIAS);
    const auto& indexMem = memory.at(ARG_SRC_1);
    const auto& dstMem = memory.at(ARG_DST);

    const auto& indexShape = indexMem->getStaticDims();
    const size_t M = indexShape[0];
    const size_t indices_size = indexShape[1];

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto bias_offset = OffsetHelper::createOffsetHelper(biasMem);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    const size_t gather_axis_size = m_weightsMemory->getStaticDims()[0];

    auto setGatherArgs = [&](dnnl_primitive_args& args, size_t gather_axis_index) {
        if (args.count(DNNL_ARG_BIAS) != 0U) {
            args[DNNL_ARG_BIAS].set_data_handle(bias_offset(gather_axis_index));
        }
        if (args.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS) != 0U) {
            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS].set_data_handle(scale_offset(gather_axis_index));
        }
        if (args.count(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS) != 0U) {
            args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS].set_data_handle(zp_offset(gather_axis_index));
        }
    };

    if (M > 1) {
        std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
        std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
        for (size_t m = 0; m < M; m++) {
            const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
            for (size_t i = 0; i < indices_size; i++) {
                int32_t gather_axis_index = gather_ids[i];
                OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                                "Invalid gather_id ",
                                gather_axis_index,
                                " for m ",
                                m);
                auto& index = elements_per_gather_indx[gather_axis_index];
                gather_idx_map[gather_axis_index * M + index] = {m, i};
                index++;
            }
        }

        if (m_bf16AmxMode) {
            OPENVINO_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
            OPENVINO_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
            OPENVINO_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");
            OPENVINO_ASSERT(m_gemmPrim, "GEMM primitive is not created");

            const auto element_size = m_tmpInputDesc->getPrecision().size();
            const auto K_size = m_tmpInputDesc->getShape().getStaticDims()[1];
            const auto M_size = m_tmpInputDesc->getShape().getStaticDims()[0];
            const auto N_size = dstMem->getStaticDims()[2];

            auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
            auto* output_ptr = input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);

            Memory tmpInput(m_context->getEngine(), m_tmpInputDesc, input_ptr);
            Memory tmpOutput(m_context->getEngine(), m_tmpOutputDesc, output_ptr);

            auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
            auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
                if (0 == num_valid_rows) {
                    continue;
                }

                cpu_parallel->parallel_for(M_size, [&](size_t m) {
                    auto* dst_row = tmp_input_offset(m);
                    if (m < num_valid_rows) {
                        const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                        const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                        std::memcpy(dst_row, src_offset(batch_index, row_id), K_size * element_size);
                    } else {
                        std::memset(dst_row, 0, K_size * element_size);
                    }
                });

                m_gemmArgs[DNNL_ARG_SRC].set_data_handle(tmp_input_offset.get_base());
                m_gemmArgs[DNNL_ARG_DST].set_data_handle(tmp_dst_offset.get_base());
                m_gemmArgs[DNNL_ARG_WEIGHTS].set_data_handle(wei_offset(gather_axis_index));
                setGatherArgs(m_gemmArgs, gather_axis_index);
                m_gemmPrim->execute(m_gemmArgs);

                cpu_parallel->parallel_for(num_valid_rows, [&](size_t m) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    std::memcpy(dst_offset(batch_index, row_id), tmp_dst_offset(m), N_size * element_size);
                });
            }
        } else {
            OPENVINO_ASSERT(m_gemvPrim, "GEMV primitive is not created");
            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                if (0 == elements_per_gather_indx[gather_axis_index]) {
                    continue;
                }
                m_gemvArgs[DNNL_ARG_WEIGHTS].set_data_handle(wei_offset(gather_axis_index));
                setGatherArgs(m_gemvArgs, gather_axis_index);
                for (int32_t m = 0; m < elements_per_gather_indx[gather_axis_index]; ++m) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    m_gemvArgs[DNNL_ARG_SRC].set_data_handle(src_offset(batch_index, row_id));
                    m_gemvArgs[DNNL_ARG_DST].set_data_handle(dst_offset(batch_index, row_id));
                    m_gemvPrim->execute(m_gemvArgs);
                }
            }
        }
    } else {
        OPENVINO_ASSERT(m_gemvPrim, "GEMV primitive is not created");

        constexpr size_t m = 0;
        auto* gather_ids = static_cast<int32_t*>(index_offset(m));
        for (size_t i = 0; i < indices_size; i++) {
            int32_t gather_axis_index = gather_ids[i];
            OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                            "Invalid gather_id ",
                            gather_axis_index,
                            " for i ",
                            i);
            m_gemvArgs[DNNL_ARG_SRC].set_data_handle(src_offset(i, m));
            m_gemvArgs[DNNL_ARG_DST].set_data_handle(dst_offset(i, m));
            m_gemvArgs[DNNL_ARG_WEIGHTS].set_data_handle(wei_offset(gather_axis_index));
            setGatherArgs(m_gemvArgs, gather_axis_index);
            m_gemvPrim->execute(m_gemvArgs);
        }
    }
}

impl_desc_type GatherMatmulDnnlExecutor::implType() const {
    return m_implType;
}

}  // namespace ov::intel_cpu
