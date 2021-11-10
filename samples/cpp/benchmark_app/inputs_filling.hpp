// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <string>
#include <vector>

#include "utils.hpp"

std::map<std::string, std::vector<InferenceEngine::Blob::Ptr>> getBlobs(
    std::map<std::string, std::vector<std::string>>& inputFiles,
    std::vector<benchmark_app::InputsInfo>& app_inputs_info);

std::map<std::string, std::vector<InferenceEngine::Blob::Ptr>> getBlobsStaticCase(
    const std::vector<std::string>& inputFiles,
    const size_t& batchSize,
    benchmark_app::InputsInfo& app_inputs_info,
    size_t requestsNum);
