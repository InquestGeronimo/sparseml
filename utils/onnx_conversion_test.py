"""
Remark 1: pruned_quant-moderate model doesn't work
"""

import os

import onnx
import onnx_graphsurgeon as gs
import torch
from sparseml.pytorch.models.classification import resnet50
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter
from sparsezoo import Zoo
from sparsezoo.models.classification import resnet_50 as zoo_resnet_50

def compare_onnx_node_list(n1, n2, key):
    if key == 'inputs':
        x1, x2 = n1.inputs, n2.inputs
    if key == 'outputs':
        x1, x2 = n1.outputs, n2.outputs

    if len(x1) != len(x2):
        raise ValueError(f"There is uneven number of {key} for node {n1.name}")

def compare_onnx_nodes(n1, n2):
    if not n1.op == n2.op:
        raise ValueError(f"Op type for node name {n1.name} is not consistent across nodes.")
    if not n1.attrs == n2.attrs:
        raise ValueError(f"Attribute for node name {n1.name} is not consistent across nodes.")
    compare_onnx_node_list(n1, n2, key='inputs')
    #compare_onnx_node_list(n1.outputs, n2.outputs, key = 'outputs')

def compare_onnx_graphs(original_onnx_file, retrieved_onnx_file):
    # We specify names of nodes, which, when comparing two onnx files,
    # could potentially be absent in one of the onnx models,
    # e.g. due to different "node folding".
    optional_nodes = ['Unsqueeze', 'Constant', 'Reshape', 'Shape', 'Gather', 'Concat']

    # Load onnx models as graph-surgeon objects.
    graph = gs.import_onnx(onnx.load(original_onnx_file))
    nodes = graph.nodes

    graph_retrieved = gs.import_onnx(onnx.load(retrieved_onnx_file))
    nodes_retrieved = graph_retrieved.nodes

    # Represent names of nodes as sets.
    s1 = {node.name for node in nodes}
    s2 = {node.name for node in nodes_retrieved}

    # Identify node set differences to investigate the unmatching nodes.
    # Also, keep only op names (e.g. `Gemm_278` --> `Gemm`)
    s1_diff_s2 = {x.split('_')[0] for x in s1.difference(s2)}
    s2_diff_s1 = {x.split('_')[0] for x in s2.difference(s1)}

    """
    Now, we can investigate leftover nodes. E.g. if
    s1_diff_s2 = {'Gemm', 'Shape', 'Reshape', 'Unsqueeze', 'Constant', 'Gather', 'Softmax', 'Concat'}
    s2_diff_s2 = {'Reshape', 'Softmax', 'Gemm'}
    
    We disregard any nodes which belong to `optional nodes`:
    s1_diff_s2` = {'Gemm','Softmax'}
    s2_diff_s2` = {'Softmax', 'Gemm'}
    and make sure that the resulting sets are equal.
    """
    s1_diff_s2 = {x for x in s1_diff_s2 if x not in optional_nodes}
    s2_diff_s1 = {x for x in s2_diff_s1 if x not in optional_nodes}

    assert s1_diff_s2 == s2_diff_s1

    # Finally, for the remaining node names, we make sure that their nodes are equal
    # for both models.
    for node in nodes:
        if node.name in s1.intersection(s2):
            node_match_found = False
            # ugly nested loop, potentially to optimize
            for node_retrieved in nodes_retrieved:
                if node_retrieved.name == node.name:
                    compare_onnx_nodes(node, node_retrieved)
                    node_match_found = True
            assert node_match_found


def run(path_retrieved_onnx, model_idx, device):
    # Fetch the model from SparseZoo
    sparse_models = Zoo.search_sparse_models(zoo_resnet_50())
    model_stub = sparse_models[model_idx]
    model_stub.download()
    path_onnx = model_stub.onnx_file.downloaded_path()

    if not os.path.isfile(path_retrieved_onnx):

        path_pth = os.path.join(os.path.dirname(path_onnx), 'pytorch', 'model.pth')
        path_recipe = os.path.join(os.path.dirname(path_onnx), 'recipes', 'original.md')

        # Load PyTorch model and export "retrieved" onnx model
        model = resnet50()
        model.eval()
        ScheduledModifierManager.from_yaml(path_recipe).apply(model)
        state_dict = torch.load(path_pth, map_location=torch.device(device))['state_dict']
        model.load_state_dict(state_dict)

        exporter = ModuleExporter(model, output_dir='')
        exporter.export_onnx(torch.randn(1, 3, 224, 224), name=path_retrieved_onnx)

    # Test 'tegridy of retrieved_onnx
    compare_onnx_graphs(path_onnx, path_retrieved_onnx)


if __name__ == "__main__":
    device = 'cpu'
    path_retrieved_onnx = 'retrieved_model.onnx'
    model_idx = 4  # picking pruned95-none 0
    run(path_retrieved_onnx, model_idx, device)
