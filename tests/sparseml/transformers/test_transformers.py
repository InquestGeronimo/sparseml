# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import onnx
import pytest

from sparsezoo import Zoo
from src.sparseml.transformers import export_transformer_to_onnx


def _compare_onnx_models(model1, model2):
    optional_nodes_model1 = [
        "If",
        "Equal",
        "Gather",
        "Shape",
        # above, those are the ops which are used in the
        # original graph to create logits and softmax heads
        "Constant",
        "Cast",
    ]  # above, the remaining optional nodes
    optional_nodes_model2 = [
        "Constant",
        "Squeeze",
    ]  # above, those are the ops which are
       # used in the original graph to create
       # logits and softmax heads

    nodes1 = model1.graph.node
    nodes1_names = [node.name for node in nodes1]
    nodes2 = model2.graph.node
    nodes2_names = [node.name for node in nodes2]

    nodes1_names_diff = [
        node_name.split("_")[0]
        for node_name in nodes1_names
        if node_name not in nodes2_names
    ]

    nodes2_names_diff = [
        node_name.split("_")[0]
        for node_name in nodes2_names
        if node_name not in nodes1_names
    ]

    assert not [x for x in nodes1_names_diff if x not in optional_nodes_model1]
    assert not [x for x in nodes2_names_diff if x not in optional_nodes_model2]

    for node1 in nodes1:
        if node1.name in set(nodes1_names).intersection(set(nodes2_names)):
            for node2 in nodes2:
                if node1.name == node2.name:
                    _compare_onnx_nodes(node1, node2)


def _compare_onnx_nodes(n1, n2):
    # checking for consistent lens seems like a sufficient test.
    # due to internal structure, the naming of connected graph nodes
    # may vary, even thought the semantics remain unchanged.
    assert len(n1.input) == len(n2.input)
    assert len(n1.output) == len(n2.output)
    assert len(n1.op_type) == len(n2.op_type)
    assert len(n1.attribute) == len(n2.attribute)


@pytest.mark.parametrize(
    "model_stub, recipe_present, task",
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-conservative",  # noqa: E501
            False,
            "question-answering",
        )
    ],
    scope="function",
)
class TestOnnxExport:
    # since we are testing multiple consecutive functionalities
    # decided to follow the structure similar to tests for pruning modifiers
    # (test_lifecycle which encapsulates granular tests)
    def test_lifecycle(self, model_stub, task, recipe_present):
        model = Zoo.load_model_from_stub(model_stub)
        model.download()

        path_onnx = model.onnx_file.downloaded_path()
        model_path = os.path.join(os.path.dirname(path_onnx), "pytorch")
        path_retrieved_onnx = "retrieved_model.onnx"

        def _test_export_transformer_to_onnx(model_path, path_retrieved_onnx, task):
            path_retrieved_onnx = export_transformer_to_onnx(
                task=task,
                model_path=model_path,
                onnx_file_name=path_retrieved_onnx,
            )
            assert onnx.load(path_retrieved_onnx)

        def _test_assert_yaml_recipe_present(model_path, recipe_present):
            assert recipe_present == any(
                [
                    file
                    for file in glob.glob(os.path.join(model_path, "*"))
                    if file.endswith(".yaml")
                ]
            )

        _test_export_transformer_to_onnx(model_path, path_retrieved_onnx, task)
        _test_assert_yaml_recipe_present(model_path, recipe_present)
        _compare_onnx_models(
            onnx.load(path_onnx),
            onnx.load(os.path.join(model_path, path_retrieved_onnx)),
        )
