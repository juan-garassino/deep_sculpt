import torch

from deepsculpt.core.data.generation.pytorch_shapes import attach_pipe_pytorch, attach_plane_pytorch


def test_attach_plane_pytorch_honors_orientation():
    structure = torch.zeros((16, 16, 16), dtype=torch.int8)
    colors = torch.zeros((16, 16, 16), dtype=torch.int16)

    structure, colors = attach_plane_pytorch(
        structure,
        colors,
        element_plane_min_ratio=0.3,
        element_plane_max_ratio=0.3,
        orientation="xy",
        device="cpu",
    )

    filled_z = (structure.sum(dim=(0, 1)) > 0).nonzero(as_tuple=False)
    assert len(filled_z) == 1


def test_attach_pipe_pytorch_accepts_axis_selection():
    structure = torch.zeros((16, 16, 16), dtype=torch.int8)
    colors = torch.zeros((16, 16, 16), dtype=torch.int16)

    structure, colors = attach_pipe_pytorch(
        structure,
        colors,
        element_volume_min_ratio=0.3,
        element_volume_max_ratio=0.3,
        axis_selection=1,
        device="cpu",
    )

    assert torch.count_nonzero(structure) > 0
