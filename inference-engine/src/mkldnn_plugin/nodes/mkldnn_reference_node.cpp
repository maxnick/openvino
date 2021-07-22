// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reference_node.h"
#include <ie_ngraph_utils.hpp>
#include <mkldnn_extension_utils.h>
#include <ngraph/runtime/host_tensor.hpp>
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

MKLDNNReferenceNode::MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache,
                                         const std::string& errorMessage) :
        MKLDNNNode(op, eng, cache), ngraphOp(op), additionalErrorMessage(errorMessage) {
    if (!op->has_evaluate()) {
        IE_THROW(NotImplemented) << "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented)";
    }
    setType(Reference);
    setTypeStr("Reference");

    if (isDynamicNode()) {
        for (size_t i = 0; i < inputShapes.size(); i++) {
            inputsForShapeInfer.push_back(std::make_shared<ngraph::opset1::Parameter>(ngraphOp->get_input_element_type(i),
                                                                                      ngraphOp->get_input_partial_shape(i)));
        }
    }
}

void MKLDNNReferenceNode::getSupportedDescriptors() {}

void MKLDNNReferenceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ngraphOp->get_input_element_type(i)), inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ngraphOp->get_output_element_type(i)), outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void MKLDNNReferenceNode::createPrimitive() {}

std::vector<std::vector<size_t>> MKLDNNReferenceNode::shapeInfer() const {
    for (size_t i = 0; i < inputsForShapeInfer.size(); i++) {
        static_cast<ngraph::opset1::Parameter *>(inputsForShapeInfer[i].get_node())->set_partial_shape(
            getParentEdgesAtPort(i)[0]->getMemory().GetDesc().getShape().toPartialShape());
        inputsForShapeInfer[i].get_node()->validate_and_infer_types();
    }

    const auto opWithNewOutputShape = ngraphOp->clone_with_new_inputs(inputsForShapeInfer);
    opWithNewOutputShape->validate_and_infer_types();

    IE_ASSERT(opWithNewOutputShape->get_output_size() == getOriginalOutputsNumber());

    std::vector<std::vector<size_t>> newShapes(getOriginalOutputsNumber());
    for (size_t i = 0; i < newShapes.size(); i++) {
        newShapes[i] = opWithNewOutputShape->get_output_partial_shape(i).get_shape();
    }
    return newShapes;
}

void MKLDNNReferenceNode::execute(mkldnn::stream strm) {
    ngraph::HostTensorVector inputs;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        void *srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().GetPtr();
        inputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphOp->get_input_element_type(i),
                                                              getParentEdgesAtPort(i)[0]->getMemory().GetDesc().getShape().getStaticDims(), srcDataPtr));
    }

    ngraph::HostTensorVector outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().GetPtr();
        outputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphOp->get_output_element_type(i),
                                                               getChildEdgesAtPort(i)[0]->getMemory().GetDesc().getShape().getStaticDims(), dstDataPtr));
    }

    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
}

void MKLDNNReferenceNode::executeDynamicBody(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNReferenceNode::created() const {
    return getType() == Reference;
}
