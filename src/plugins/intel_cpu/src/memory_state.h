// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "blob_factory.hpp"
#include "cpu_memory.h"
#include "nodes/common/cpu_memcpy.h"
#include "memory_desc/cpu_memory_desc_utils.h"

#include <string>

namespace ov {
namespace intel_cpu {

class VariableState : public InferenceEngine::IVariableStateInternal {
public:
    VariableState(std::string name, MemoryPtr storage)
        : InferenceEngine::IVariableStateInternal{name} {
        tensor_desc = MemoryDescUtils::convertToTensorDesc(storage->getDesc());
    }

    void Reset() override;

private:
    InferenceEngine::TensorDesc tensor_desc;  // shape of initial state
};

}   // namespace intel_cpu
}   // namespace ov
