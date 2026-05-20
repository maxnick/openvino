// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class GatherMatmulDnnlExecutor : public Executor {
public:
    static bool supports(const GatherMatmulConfig& config);

    GatherMatmulDnnlExecutor(const GatherMatmulAttrs& attrs,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    ExecutorContext::CPtr m_context;

    // Packed weight / scale / ZP tensors with the gather batch dimension prepended
    MemoryPtr m_weightsMemory;
    MemoryPtr m_scalesMemory;
    MemoryPtr m_zpMemory;

    // GEMV primitive (M=1) — created once in constructor, used for all M=1 calls
    // and as the M>1 non-AMX fallback
    DnnlFCPrimitivePtr m_gemvPrim;
    MemoryPtr m_gemvScratchpad;      // fixed allocation for GEMV scratchpad
    dnnl_primitive_args m_gemvArgs;  // pre-built args map, handles updated per-call

    // GEMM primitive (M>1, AMX bf16 path) — re-created in update() when M changes
    DnnlFCPrimitivePtr m_gemmPrim;
    MemoryPtr m_gemmScratchpad;      // embedded in m_tmpInpBuffer
    dnnl_primitive_args m_gemmArgs;  // pre-built args map, handles updated per-call

    // Temporary pack/scatter buffers for the AMX GEMM path (allocated in update())
    MemoryPtr m_tmpInpBuffer;
    MemoryDescPtr m_tmpInputDesc;
    MemoryDescPtr m_tmpOutputDesc;

    bool m_bf16AmxMode = false;
    impl_desc_type m_implType = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
