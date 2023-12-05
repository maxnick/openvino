// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <nodes/common/cpu_convert.h>

#include "memory_state.h"

#include "dnnl_extension_utils.h"
#include "blob_factory.hpp"
#include "cpu_tensor.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

VariableStateBase::VariableStateBase(const std::string& name, const MemoryDescPtr& external_desc) :
    IVariableState{name} , m_external_desc{external_desc} {}

MemoryDescPtr VariableStateBase::to_static(const MemoryDescPtr& desc) {
    if (!desc->isDefined()) {
        auto&& current_dims = desc->getShape().getDims();
        VectorDims new_dims(current_dims.size());
        std::transform(current_dims.begin(), current_dims.end(), new_dims.begin(), [](Dim x) {
            return x == Shape::UNDEFINED_DIM ? 0 : x; });

        return desc->cloneWithNewDims(new_dims, true);
    }
    return desc;
}

const dnnl::engine& VariableStateBase::get_engine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

void VariableStateBase::set_state_impl(const ov::SoPtr<ov::ITensor>& state) {
    m_state = state; // simply to extend the lifetime
    auto state_desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(m_state);

    const auto& shape = state_desc->getShape();

    if (input_mem()->getShape() != shape) {
        auto new_desc = internal_desc()->cloneWithNewDims(shape.getStaticDims());
        input_mem()->redefineDesc(new_desc);
    }

    auto src = m_state->data();

    Memory mem(get_engine(), state_desc, src);
    input_mem()->load(mem);
}

void VariableStateBase::set_state(const ov::SoPtr<ov::ITensor>& state) {
    set_state_impl(state);
    reset_state_flag = false;
}

ov::SoPtr<ov::ITensor> VariableStateBase::get_state() const {
    const auto& current_dims = internal_state_mem()->getStaticDims();
    auto current_ext_desc = m_external_desc->cloneWithNewDims(current_dims);
    auto current_internal_desc = internal_state_mem()->getDescPtr();

    if (current_ext_desc->isCompatible(*current_internal_desc)) {
        return std::make_shared<Tensor>(internal_state_mem());
    }

    //test precision
    {
        auto internal_prc = current_internal_desc->getPrecision();
        auto tmp_desc = current_ext_desc->cloneWithNewPrecision(internal_prc);
        if (tmp_desc->isCompatible(*current_internal_desc)) {
            auto mem = std::make_shared<Memory>(get_engine(), current_ext_desc);
            size_t elements_to_convert = internal_state_mem()->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            auto external_prc = current_ext_desc->getPrecision();

            cpu_convert(internal_state_mem()->getData(), mem->getData(), internal_prc, external_prc, elements_to_convert);
            return std::make_shared<Tensor>(mem);
        }
    }

    //reorder
    auto mem = std::make_shared<Memory>(get_engine(), current_ext_desc);
    mem->load(*(internal_state_mem()));
    return std::make_shared<Tensor>(mem);
}

void VariableStateBase::reset() {
    reset_impl();
    reset_state_flag = true;
}

bool VariableStateBase::is_reset_state() const {
    return reset_state_flag;
}

void VariableStateBase::commit() {
    commit_impl();
    reset_state_flag = false;
}

VariableStateDoubleBuffer::VariableStateDoubleBuffer(const std::string& name,
                                                     const MemoryPtr& first_buffer,
                                                     const MemoryPtr& second_buffer,
                                                     const MemoryDescPtr& external_desc) :
    VariableStateBase(name, external_desc) {
    OPENVINO_ASSERT(first_buffer && second_buffer);
    reset_prime_mem(first_buffer);
    reset_second_mem(second_buffer);
    m_internal_desc = prime_mem()->getDescPtr();
    auto&& shape = m_internal_desc->getShape();
    //TODO what if by some reason we already have internal static state while the node is dynamic, is it even possible?

    if (shape.isStatic()) {
        prime_mem()->nullify();
    } else {
        //in the case of the original desc has dynamic shape we create an empty tensor
        auto new_desc = to_static(m_internal_desc);
        prime_mem()->redefineDesc(new_desc);
    }
}

void VariableStateDoubleBuffer::reset_impl() {
    auto new_desc = to_static(m_internal_desc);
    for (auto&& mem : m_internal_mem) {
        if (mem) {
            mem->redefineDesc(new_desc);
            mem->nullify();
        }
    }
}

void VariableStateDoubleBuffer::commit_impl() {
    buffer_num ^= 0x01;
}

MemoryPtr VariableStateDoubleBuffer::input_mem() {
    return prime_mem();
}

MemoryPtr VariableStateDoubleBuffer::output_mem() {
    return second_mem();
}

MemoryDescPtr VariableStateDoubleBuffer::internal_desc() const {
    return m_internal_desc;
}

MemoryPtr VariableStateDoubleBuffer::internal_state_mem() const {
    return prime_mem();
}

VariableStateSingleBuffer::VariableStateSingleBuffer(const std::string& name,
                                                     const MemoryPtr& buffer,
                                                     const MemoryDescPtr& external_desc) :
    VariableStateBase(name, external_desc) {
    OPENVINO_ASSERT(buffer);
    m_internal_mem = buffer;
    m_internal_desc = m_internal_mem->getDescPtr();
    auto&& shape = m_internal_desc->getShape();
    //TODO what if by some reason we already have internal static state while the node is dynamic, is it even possible?

    if (shape.isStatic()) {
        m_internal_mem->nullify();
    } else {
        //in the case of the original desc has dynamic shape we create an empty tensor
        auto new_desc = to_static(m_internal_desc);
        m_internal_mem->redefineDesc(new_desc);
    }
}

void VariableStateSingleBuffer::reset_impl() {
    auto new_desc = to_static(m_internal_desc);
    m_internal_mem->redefineDesc(new_desc);
    m_internal_mem->nullify();
}

MemoryPtr VariableStateSingleBuffer::input_mem() {
    return m_internal_mem;
}

MemoryPtr VariableStateSingleBuffer::output_mem() {
    return m_internal_mem;
}

MemoryDescPtr VariableStateSingleBuffer::internal_desc() const {
    return m_internal_desc;
}

MemoryPtr VariableStateSingleBuffer::internal_state_mem() const {
    return m_internal_mem;
}

void VariableStateSingleBuffer::commit_impl() {
    //nothing to do
}


VariableStateKVcache::VariableStateKVcache(const std::string& name, const MemoryDescPtr& external_desc) :
    VariableStateBase(name, external_desc) {
    auto&& shape = external_desc->getShape();

    OPENVINO_ASSERT(shape.isDynamic(), "VariableStateKVcache is unexpectedly initalized with a static tensor");
}

ov::SoPtr<ov::ITensor> VariableStateKVcache::get_state() const {
    //TBD
    return {};
}

void VariableStateKVcache::set_state_impl(const ov::SoPtr<ov::ITensor>& state) {
    //1. reset the memory object
    m_state = state; // simply to extend the lifetime
    auto state_desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(m_state);

    //May be optimized by reusing the state tensor underlining memory pointer, but corner cases should be considered
    m_internal_mem = std::make_shared<Memory>(get_engine(), state_desc);
    auto src = m_state->data();
    auto dst = m_internal_mem->getData();
    if (src && dst) {
        std::memcpy(dst, src, m_state->get_byte_size());
    }

    //2. Reset the beam search table
    auto&& stateDims = state_desc->getShape().getStaticDims();
    const size_t size_B = stateDims[axis_B];
    const size_t size_L = stateDims[axis_L];
    auto mem_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, Shape{size_B, size_L});

    m_hidden_state = std::make_shared<Memory>(get_engine(), mem_desc);
    auto buff = reinterpret_cast<int*>(m_hidden_state->getData());
    for (size_t i = 0; i < size_B; ++i) {
        for (size_t j = 0; j < size_L; ++j) {
            buff[i * size_B + j] = i;
        }
    }
}

void VariableStateKVcache::reset_impl() {
    // 1. reset internal state
    auto internal_state_desc = to_static(get_external_desc());
    m_internal_mem = std::make_shared<Memory>(get_engine(), internal_state_desc);
    m_internal_mem->nullify();

    // 2. reset hidden state
    auto&& stateDims = internal_state_desc->getShape().getStaticDims();
    const size_t size_B = stateDims[axis_B];
    const size_t size_L = stateDims[axis_L];
    auto hidden_state_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, Shape{size_B, size_L});

    m_hidden_state = std::make_shared<Memory>(get_engine(), hidden_state_desc);
    m_hidden_state->nullify();
}

void VariableStateKVcache::commit_impl() {
    //nothing to do
}

MemoryPtr VariableStateKVcache::input_mem() {
    return m_internal_mem;
}

MemoryPtr VariableStateKVcache::output_mem() {
    return m_internal_mem;
}

MemoryDescPtr VariableStateKVcache::internal_desc() const {
    return m_internal_mem->getDescPtr(); //since we don't store initial one
}

MemoryPtr VariableStateKVcache::internal_state_mem() const {
    return m_internal_mem;
}

void VariableStateKVcache::assign_internal_state(const MemoryPtr& mem) {
    m_internal_mem = mem;
}

MemoryPtr VariableStateKVcache::hidden_state_mem() const {
    return m_hidden_state;
}

void VariableStateKVcache::assign_hidden_state(const MemoryPtr& mem) {
    m_hidden_state = mem;
}

}  // namespace intel_cpu
}  // namespace ov
