// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/executable_network.hpp"

#include <pybind11/stl.h>

#include "common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/infer_request.hpp"

PYBIND11_MAKE_OPAQUE(Containers::TensorIndexMap);
PYBIND11_MAKE_OPAQUE(Containers::TensorNameMap);

namespace py = pybind11;

void regclass_ExecutableNetwork(py::module m) {
    py::class_<ov::runtime::ExecutableNetwork, std::shared_ptr<ov::runtime::ExecutableNetwork>> cls(
        m,
        "ExecutableNetwork");

    cls.def(py::init([](ov::runtime::ExecutableNetwork& other) {
                return other;
            }),
            py::arg("other"));

    cls.def("create_infer_request", [](ov::runtime::ExecutableNetwork& self) {
        return InferRequestWrapper(self.create_infer_request(), self.inputs(), self.outputs());
    });

    cls.def(
        "infer_new_request",
        [](ov::runtime::ExecutableNetwork& self, const py::dict& inputs) {
            auto request = self.create_infer_request();
            // Update inputs if there are any
            Common::set_request_tensors(request, inputs);
            request.infer();
            return Common::outputs_to_dict(self.outputs(), request);
        },
        py::arg("inputs"));

    cls.def("export_model", &ov::runtime::ExecutableNetwork::export_model, py::arg("network_model"));

    cls.def(
        "get_config",
        [](ov::runtime::ExecutableNetwork& self, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_config(name)).as<py::object>();
        },
        py::arg("name"));

    cls.def(
        "get_metric",
        [](ov::runtime::ExecutableNetwork& self, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_metric(name)).as<py::object>();
        },
        py::arg("name"));

    cls.def("get_runtime_function", &ov::runtime::ExecutableNetwork::get_runtime_function);

    cls.def_property_readonly("inputs", &ov::runtime::ExecutableNetwork::inputs);

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)() const) &
                ov::runtime::ExecutableNetwork::input);

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(size_t) const) &
                ov::runtime::ExecutableNetwork::input,
            py::arg("i"));

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(const std::string&) const) &
                ov::runtime::ExecutableNetwork::input,
            py::arg("tensor_name"));

    cls.def_property_readonly("outputs", &ov::runtime::ExecutableNetwork::outputs);

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)() const) &
                ov::runtime::ExecutableNetwork::output);

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(size_t) const) &
                ov::runtime::ExecutableNetwork::output,
            py::arg("i"));

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(const std::string&) const) &
                ov::runtime::ExecutableNetwork::output,
            py::arg("tensor_name"));
}
