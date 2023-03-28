// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include "cpu_shape.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/node_config.h"
#include "weights_cache.hpp"

#include <map>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {

class Node;
class Edge;

using EdgePtr = std::shared_ptr<Edge>;
using EdgeWeakPtr = std::weak_ptr<Edge>;
using EdgeRawPtr = Edge*;
using EdgeRawCPtr = const Edge*;

class Edge {
public:
    Edge(const std::shared_ptr<Node>& parent,
         const std::shared_ptr<Node>& child,
         int pr_port = 0, int ch_port = 0);

    Edge(const Edge&) = delete;
    Edge(Edge&&) = delete;
    Edge& operator=(const Edge&) = delete;
    Edge& operator=(Edge&&) = delete;
    ~Edge();

    enum class Status {
        Uninitialized,
        NeedAllocation,
        NotAllocated,
        Allocated,
        Validated
    };

    enum class ReorderStatus {
        Regular = 0,
        Optimized = 1,
        No = 2
    };

    inline Status getStatus() const noexcept {
        return status;
    }

    void changeStatus(Status state);

    void init();
    void allocate(const void* mem_ptr = nullptr);
    void allocate(DnnlMemoryMngrPtr memMngr);
    void externalAllocate(WeightsSharing::Ptr weightsCache);
    void reuse(MemoryPtr ptr);
    void validate();
    void drop();

    const std::shared_ptr<Node> getParent() const;
    const std::shared_ptr<Node> getChild() const;

    const Memory& getMemory();
    MemoryPtr& getMemoryPtr();

    ReorderStatus needReorder();
    bool isDropped() const;
    bool isUseExternalMemory() const;

    int getInputNum() const;
    int getOutputNum() const;

    void setChildPort(const size_t port) { child_port = port; }

    void sharedMemFrom(EdgeRawPtr edge);
    EdgeRawPtr getSharedEdge() const;
    EdgeRawPtr getSharedEdge(std::nothrow_t) const;

    bool hasDefinedMaxSize() const {
        return getDesc().hasDefinedMaxSize();
    }

    std::string name() const;

private:
    std::weak_ptr<Node> parent;
    std::weak_ptr<Node> child;
    int parent_port;
    int child_port;

    bool useExternalMemory = false;
    EdgeRawPtr memoryFromEdge = nullptr;
    std::unordered_set<EdgeRawPtr> mem_consumers;
    MemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    const MemoryDesc& getInputDesc() const;
    const MemoryDesc& getOutputDesc() const;
    PortDescBaseCPtr getInputPortDesc() const;
    PortDescBaseCPtr getOutputPortDesc() const;
    void register_as_consumer(EdgeRawPtr edge);
    void deregister_mem_provider(EdgeRawPtr edge);

    const MemoryDesc& getDesc() const;
    bool enforceReorder();

    void collectConsumers(std::vector<std::shared_ptr<Node>>& result) const;

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN, LOOK_NO_RECURRENT = 4 };

    EdgeRawPtr getBaseEdge(int look = LOOK_BOTH);
    bool inPlace(LOOK look = LOOK_BOTH) const;
    void allocateCommon(const std::function<void(const MemoryPtr&, const MemoryDesc&)>& allocate);

    friend class Graph;
};

}   // namespace intel_cpu
}   // namespace ov

