// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_dot_products(void** a, void** b, void**c, size_t vec_num, size_t vec_len, ov::element::Type precision_a, ov::element::Type precision_b);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine