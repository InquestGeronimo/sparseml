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
from collections import Counter

import onnx
import pytest

from sparsezoo import Zoo
from src.sparseml.transformers import export_transformer_to_onnx


def _compare_onnx_nodes(n1, n2, all_node_ids):
    if n1.input != n2.input:
        # Constant Op
        if not n1.input and not n2.input:
            print(
                f"Error: {n1.name}. "
                f"Input n1 : {n1.input}, "
                f"but input n2: {n2.input}"
            )
        elif (not n1.input[0] in all_node_ids) or (not n2.input[0] in all_node_ids):
            print(
                f"Error: {n1.name}. "
                f"Input n1 : {n1.input}, "
                f"but input n2: {n2.input}"
            )
    if (
        n1.output != n2.output
        and (not n1.output[0] in all_node_ids)
        or (not n2.output[0] in all_node_ids)
    ):
        print(
            f"Error: {n1.name}. "
            f"Output n1 : {n1.output}, "
            f"but output n2: {n2.output}"
        )
    if n1.op_type != n2.op_type:
        print(
            f"Error: {n1.name}. "
            f"Op type n1 : {n1.op_type}, "
            f"but op type n2: {n2.op_type}"
        )
    if n1.attribute != n2.attribute:
        print(
            f"Error: {n1.name}. "
            f"Attribute n1 : {n1.attribute}, "
            f"but attribute n2: {n2.attribute}"
        )


def _compare_onnx_models(model1, model2):

    nodes1 = model1.graph.node
    nodes2 = model2.graph.node

    nodes1_names = [node.name for node in nodes1]
    nodes2_names = [node.name for node in nodes2]

    # Find the nodes in nodes1,
    # which are not in nodes2 and save them
    # as tuples (op_name, op_num) e.g. (Add, 234)
    nodes1_names_diff_ = [
        node_name.split("_")
        for node_name in nodes1_names
        if node_name not in nodes2_names
    ]
    # Find the nodes in nodes2,
    # which are not in nodes1 and save them
    # as tuples (op_name, op_num) e.g. (Add, 234)
    nodes2_names_diff_ = [
        node_name.split("_")
        for node_name in nodes2_names
        if node_name not in nodes1_names
    ]

    counter_nodes1 = Counter([x[0] for x in nodes1_names_diff_])
    counter_nodes2 = Counter([x[0] for x in nodes2_names_diff_])

    if counter_nodes1 != counter_nodes2:
        print(f"Nodes unique for model1 and their count:\n{counter_nodes1}")

    if counter_nodes1 != counter_nodes2:
        print(f"Nodes unique for model2 and their count:\n{counter_nodes2}")

    all_node_ids = {x[1] for x in nodes1_names_diff_}.union(
        {x[1] for x in nodes2_names_diff_}
    )

    for node1 in nodes1:
        if node1.name in set(nodes1_names).intersection(set(nodes2_names)):
            for node2 in nodes2:
                if node1.name == node2.name:
                    print("----")
                    print(node1.name)
                    _compare_onnx_nodes(node1, node2, all_node_ids=all_node_ids)


@pytest.mark.parametrize(
    "model_stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_90",  # noqa: E501
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_6layers-aggressive_94",  # noqa: E501
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant_3layers-aggressive_84",  # noqa: E501
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-moderate",  # noqa: E501
    ],
    scope="function",
)
def test_export_torch_to_onnx(model_stub):
    path_retrieved_onnx = "retrieved_model.onnx"

    model = Zoo.load_model_from_stub(model_stub)
    model.download()
    path_onnx = model.onnx_file.downloaded_path()
    model_path = os.path.join(os.path.dirname(path_onnx), "pytorch")

    path_retrieved_onnx = export_transformer_to_onnx(
        task="question-answering",
        convert_qat=True,  # this may not be working
        model_path=model_path,
        onnx_file_name=path_retrieved_onnx,
    )

    print(f"Testing model:\n{os.path.basename(model_stub)}...")
    print(f"Original onnx model path:\n{path_onnx}")
    print(f"Retrieved onnx model path:\n{path_retrieved_onnx}")

    recipe_found = any(
        [
            file
            for file in glob.glob(os.path.join(model_path, "*"))
            if file.endswith(".yaml")
        ]
    )
    print(f"Recipe file found: {recipe_found}.")

    _compare_onnx_models(onnx.load(path_onnx), onnx.load(path_retrieved_onnx))
