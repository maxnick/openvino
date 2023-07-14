// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace intel_cpu {

class CompiledModel;
class AsyncInferRequest;

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    SyncInferRequest(std::shared_ptr<const CompiledModel> compiled_model);
    virtual ~SyncInferRequest();

    void infer() override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& _port) const override;

    /**
     * @brief      Sets the pointer to asynchronous inference request that holds this request
     * @param[in]  asyncRequest Pointer to asynchronous inference request
     */
    void set_async_request(AsyncInferRequest* asyncRequest);

    /**
     * @brief If `m_asyncRequest` is initialized throw exception with `ov::Cancelled` status if inference request is
     * canceled
     */
    void throw_if_canceled() const;

private:
    void create_infer_request();

    void pushInput(const std::string& inputName, ov::SoPtr<ov::ITensor>& inputBlob, InferenceEngine::Precision dataType);
    void init_tensor(const std::string& name);
    void push_input_data();

    Graph* graph = nullptr;
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> external_ptr;

    void push_states();
    void pull_states();
    void redefine_memory_for_input_nodes();

    void update_external_inputs();

    InferenceEngine::TensorDesc create_tensor_desc(const ov::SoPtr<ov::ITensor>& tensor);

    // Transformation shouldn't change model's input/output's precision, but actually it does.
    // Some additional methods will handle it.
    const ov::Output<const ov::Node>& get_internal_port(const ov::Output<const ov::Node>& port) const;
    ov::SoPtr<ov::ITensor> get_port_tensor(const ov::Output<const ov::Node>& port) const;
    // Check this port is original model port or compiled model's port, return true for compiled model's port
    // This is because if precision has been changed in compiled model, it needs distinguish them
    bool check_compiled_model_port(const ov::Output<const ov::Node>& port) const;
    mutable std::unordered_map<std::string, ov::Output<const ov::Node>> m_orig_ports_map;
    // Store external tensor due to precision changes
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_aux_tensors;
    mutable std::unordered_map<std::string, bool> m_port_precision_changed;
    bool m_is_legacy_api = false;

    std::shared_ptr<const CompiledModel> m_compiled_model;
    openvino::itt::handle_t m_profiling_task;
    std::vector<ov::SoPtr<ov::IVariableState>> m_memory_states;
    AsyncInferRequest* m_asyncRequest = nullptr;

    mutable std::unordered_map<std::string, ov::Output<const ov::Node>> m_input_ports_map;
    mutable std::unordered_map<std::string, ov::Output<const ov::Node>> m_output_ports_map;
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_outputs;

    void change_default_ptr();
};

}  // namespace intel_cpu
}  // namespace ov
