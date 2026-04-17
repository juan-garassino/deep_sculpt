
"""
PyTorch-based 3D Shape Generation System for DeepSculpt
This module provides comprehensive functionality for creating various 3D voxel-based shapes
using PyTorch tensors instead of NumPy arrays, with support for GPU acceleration and sparse tensors.
"""

import torch
import random
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional, Union
from core.utils.logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)


# ======================================================================================
# ENUMS
# ======================================================================================

class ShapeType(Enum):
    """Enumeration of different shape types."""
    EDGE = 1   # 1D linear shape
    PLANE = 2  # 2D planar shape
    PIPE = 3   # 3D hollow shape (box with empty interior)
    VOLUME = 4 # 3D solid shape (filled box)


# ======================================================================================
# SPARSE HANDLER
# ======================================================================================

class SparseTensorHandler:
    """Handles conversion between dense and sparse tensor representations."""

    @staticmethod
    def to_sparse(tensor: torch.Tensor, threshold: float = 0.01) -> torch.sparse.FloatTensor:
        """Convert dense tensor to sparse if sparsity exceeds threshold."""
        if tensor.is_sparse:
            return tensor

        sparsity = 1.0 - (torch.count_nonzero(tensor).float() / tensor.numel())
        if sparsity > threshold:
            return tensor.to_sparse()
        return tensor

    @staticmethod
    def to_dense(sparse_tensor: torch.Tensor) -> torch.Tensor:
        """Convert sparse tensor to dense."""
        if sparse_tensor.is_sparse:
            return sparse_tensor.to_dense()
        return sparse_tensor

    @staticmethod
    def detect_sparsity(tensor: torch.Tensor) -> float:
        """Calculate sparsity ratio of tensor."""
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        return 1.0 - (torch.count_nonzero(tensor).float() / tensor.numel())

    @staticmethod
    def should_use_sparse(tensor: torch.Tensor, memory_threshold: float = 0.8) -> bool:
        """Determine if sparse representation would be beneficial."""
        sparsity = SparseTensorHandler.detect_sparsity(tensor)
        return sparsity > memory_threshold


# ======================================================================================
# PYTORCH UTILS
# ======================================================================================

class PyTorchUtils:
    """PyTorch-specific utility functions for shape generation."""

    @staticmethod
    def generate_random_size(
        min_ratio: float,
        max_ratio: float,
        base_size: int,
        step: int = 1,
        device: str = "cpu",
    ) -> int:
        """Generate a random size based on given ratios and base size."""
        min_size = max(int(min_ratio * base_size), 2)
        max_size = max(int(max_ratio * base_size), min_size + 1)

        if step > 1:
            min_size = (min_size // step) * step
            max_size = (max_size // step) * step
            if min_size == max_size:
                return min_size

        return random.randrange(min_size, max_size, step)

    @staticmethod
    def select_random_position(max_pos: int, size: int) -> int:
        """Select a random position to insert a shape within bounds."""
        return random.randint(0, max(0, max_pos - size))

    @staticmethod
    def select_random_color(colors: Union[List[str], str]) -> str:
        """Select a random color from a list or return the color if it's a string."""
        if isinstance(colors, list):
            return random.choice(colors)
        return colors

    @staticmethod
    def validate_dimensions(shape_size: List[int], tensor_shape: Tuple[int, ...]) -> bool:
        """Validate that the shape fits within the tensor dimensions."""
        return all(s <= v for s, v in zip(shape_size, tensor_shape))

    @staticmethod
    def validate_bounds(
        start_pos: List[int],
        shape_size: List[int],
        tensor_shape: Tuple[int, ...],
    ) -> bool:
        """Validate that the shape at the given position fits within bounds."""
        for i in range(len(start_pos)):
            if start_pos[i] < 0 or start_pos[i] + shape_size[i] > tensor_shape[i]:
                return False
        return True


# ======================================================================================
# EDGE FUNCTIONS
# ======================================================================================

def attach_edge_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    element_edge_min_ratio: float = 0.1,
    element_edge_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    batch_size: int = 1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach an edge (1D line) to the structure using PyTorch tensors.
    """
    begin_section("Attaching edge (1D line) - PyTorch")

    try:
        if colors_dict is None:
            colors_dict = {"edges": "red"}

        structure = structure.to(device)
        colors = colors.to(device)
        original_sparse_mode = sparse_mode

        axis = random.randint(0, 2)
        log_info(f"Selected axis: {axis}")

        edge_size = PyTorchUtils.generate_random_size(
            element_edge_min_ratio,
            element_edge_max_ratio,
            structure.shape[axis],
            step,
            device,
        )
        log_info(f"Edge size: {edge_size}")

        edge_position = PyTorchUtils.select_random_position(structure.shape[axis], edge_size)
        log_info(f"Edge position on axis {axis}: {edge_position}")

        position = [slice(None), slice(None), slice(None)]
        position[axis] = slice(edge_position, edge_position + edge_size)

        other_axes = [i for i in range(3) if i != axis]
        for i in other_axes:
            position[i] = random.randint(0, structure.shape[i] - 1)

        pos_info = [
            f"{i}: {'slice' if isinstance(p, slice) else p}" for i, p in enumerate(position)
        ]
        log_info(f"Edge position indices: {pos_info}")

        edge_color = PyTorchUtils.select_random_color(colors_dict["edges"])
        log_info(f"Selected color: {edge_color}")

        if structure.is_sparse:
            structure = SparseTensorHandler.to_dense(structure)
        if colors.is_sparse:
            colors = SparseTensorHandler.to_dense(colors)

        structure[tuple(position)] = 1

        if isinstance(edge_color, str):
            color_value = hash(edge_color) % 256
        else:
            color_value = edge_color

        colors[tuple(position)] = color_value

        if original_sparse_mode and SparseTensorHandler.should_use_sparse(structure):
            structure = SparseTensorHandler.to_sparse(structure)
            colors = SparseTensorHandler.to_sparse(colors)
            log_info("Converted to sparse tensor representation")

        if not structure.is_sparse:
            structure = structure.to(torch.int8)

        log_success("Edge attached successfully")
        end_section()
        return structure, colors

    except Exception as e:
        log_error(f"Error attaching edge: {str(e)}")
        end_section("Edge attachment failed")
        raise


def attach_edges_batch_pytorch(
    structures: torch.Tensor,
    colors: torch.Tensor,
    element_edge_min_ratio: float = 0.1,
    element_edge_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach edges to multiple structures in batch using PyTorch tensors.
    """
    begin_section("Attaching edges (batch processing) - PyTorch")

    try:
        batch_size = structures.shape[0]
        log_info(f"Processing batch of {batch_size} structures")

        for i in range(batch_size):
            structures[i], colors[i] = attach_edge_pytorch(
                structures[i],
                colors[i],
                element_edge_min_ratio,
                element_edge_max_ratio,
                step,
                colors_dict,
                device,
                sparse_mode,
                1,  # batch_size = 1 for individual processing
                verbose=False,
            )

        log_success(f"Batch edge attachment completed for {batch_size} structures")
        end_section()
        return structures, colors

    except Exception as e:
        log_error(f"Error in batch edge attachment: {str(e)}")
        end_section("Batch edge attachment failed")
        raise

# ======================================================================================
# PLANE FUNCTIONS
# ======================================================================================

def return_axis_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    device: str = "cpu",
    orientation: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Selects a random plane from a 3D tensor along a random axis.
    """
    orientation_to_axis = {
        "yz": 0,
        "xz": 1,
        "xy": 2,
    }
    axis_selection = orientation_to_axis.get(orientation, random.randint(0, 2))
    section = random.randint(0, structure.shape[axis_selection] - 1)

    log_info(f"Selected axis {axis_selection}, section {section}", is_last=False)

    if axis_selection == 0:
        working_plane = structure[section, :, :]
        color_parameters = colors[section, :, :]
    elif axis_selection == 1:
        working_plane = structure[:, section, :]
        color_parameters = colors[:, section, :]
    elif axis_selection == 2:
        working_plane = structure[:, :, section]
        color_parameters = colors[:, :, section]
    else:
        log_error("Axis selection value out of range.")
        raise ValueError("Axis selection value out of range.")

    return working_plane, color_parameters, section


def attach_plane_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    element_plane_min_ratio: float = 0.1,
    element_plane_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    rotation_angle: float = 0.0,
    orientation: str = "random",  # "xy", "xz", "yz", "random"
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach a plane (2D surface) to the structure using PyTorch tensors.
    """
    begin_section("Attaching plane (2D surface) - PyTorch")

    try:
        if colors_dict is None:
            colors_dict = {"planes": "green"}

        structure = structure.to(device)
        colors = colors.to(device)
        original_sparse_mode = sparse_mode

        working_plane, color_parameters, section = return_axis_pytorch(
            structure,
            colors,
            device,
            orientation=orientation,
        )
        log_info(f"Working on plane with shape {working_plane.shape}")

        plane_width = PyTorchUtils.generate_random_size(
            element_plane_min_ratio,
            element_plane_max_ratio,
            working_plane.shape[0],
            step,
            device,
        )
        plane_height = PyTorchUtils.generate_random_size(
            element_plane_min_ratio,
            element_plane_max_ratio,
            working_plane.shape[1],
            step,
            device,
        )

        element = torch.ones((plane_width, plane_height), device=device)
        log_info(f"Created plane element with shape {element.shape}")

        delta = torch.tensor(working_plane.shape, device=device) - torch.tensor(element.shape, device=device)

        if torch.any(delta < 0):
            log_warning("Plane too large for working plane, skipping")
            end_section("Plane attachment skipped")
            return structure, colors

        top_left_corner = torch.tensor([
            random.randint(0, max(0, delta[0].item())),
            random.randint(0, max(0, delta[1].item())),
        ], device=device)

        bottom_right_corner = top_left_corner + torch.tensor(element.shape, device=device)

        log_info(f"Placing plane at position {top_left_corner.tolist()} to {bottom_right_corner.tolist()}")

        plane_color = PyTorchUtils.select_random_color(colors_dict["planes"])
        log_info(f"Selected color: {plane_color}")

        if isinstance(plane_color, str):
            color_value = hash(plane_color) % 256
        else:
            color_value = plane_color

        working_plane[
            top_left_corner[0]: bottom_right_corner[0],
            top_left_corner[1]: bottom_right_corner[1],
        ] = element

        color_parameters[
            top_left_corner[0]: bottom_right_corner[0],
            top_left_corner[1]: bottom_right_corner[1],
        ] = color_value

        if original_sparse_mode and SparseTensorHandler.should_use_sparse(structure):
            structure = SparseTensorHandler.to_sparse(structure)
            colors = SparseTensorHandler.to_sparse(colors)
            log_info("Converted to sparse tensor representation")

        if not structure.is_sparse:
            structure = structure.to(torch.int8)

        log_success("Plane attached successfully")
        end_section()
        return structure, colors

    except Exception as e:
        log_error(f"Error attaching plane: {str(e)}")
        end_section("Plane attachment failed")
        raise


def attach_plane_with_rotation_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    element_plane_min_ratio: float = 0.1,
    element_plane_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    rotation_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach a plane with 3D rotation support using PyTorch tensors.
    """
    begin_section("Attaching rotated plane (2D surface) - PyTorch")

    try:
        if colors_dict is None:
            colors_dict = {"planes": "green"}

        log_info(f"Rotation angles: {rotation_angles} (rotation not yet implemented)")

        result = attach_plane_pytorch(
            structure,
            colors,
            element_plane_min_ratio,
            element_plane_max_ratio,
            step,
            colors_dict,
            device,
            sparse_mode,
            verbose=verbose,
        )

        log_success("Rotated plane attached successfully (using basic plane for now)")
        end_section()
        return result

    except Exception as e:
        log_error(f"Error attaching rotated plane: {str(e)}")
        end_section("Rotated plane attachment failed")
        raise


def validate_plane_dimensions_pytorch(
    plane_size: Tuple[int, int],
    structure_shape: Tuple[int, int, int],
    position: Tuple[int, int, int],
    device: str = "cpu",
) -> bool:
    """
    Validate that a plane fits within the structure at the given position.
    """
    plane_tensor = torch.tensor(plane_size, device=device)
    structure_tensor = torch.tensor(structure_shape[:2], device=device)
    position_tensor = torch.tensor(position[:2], device=device)

    end_position = position_tensor + plane_tensor
    return torch.all(end_position <= structure_tensor).item()


def attach_planes_batch_pytorch(
    structures: torch.Tensor,
    colors: torch.Tensor,
    element_plane_min_ratio: float = 0.1,
    element_plane_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach planes to multiple structures in batch using PyTorch tensors.
    """
    begin_section("Attaching planes (batch processing) - PyTorch")

    try:
        batch_size = structures.shape[0]
        log_info(f"Processing batch of {batch_size} structures")

        for i in range(batch_size):
            structures[i], colors[i] = attach_plane_pytorch(
                structures[i],
                colors[i],
                element_plane_min_ratio,
                element_plane_max_ratio,
                step,
                colors_dict,
                device,
                sparse_mode,
                verbose=False,
            )

        log_success(f"Batch plane attachment completed for {batch_size} structures")
        end_section()
        return structures, colors

    except Exception as e:
        log_error(f"Error in batch plane attachment: {str(e)}")
        end_section("Batch plane attachment failed")
        raise

# ======================================================================================
# PIPE FUNCTIONS
# ======================================================================================

def attach_pipe_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    element_volume_min_ratio: float = 0.1,
    element_volume_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    wall_thickness: int = 1,
    pipe_complexity: str = "simple",  # "simple", "complex", "curved"
    axis_selection: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach a pipe (hollow 3D structure) to the structure using PyTorch tensors.
    """
    begin_section("Attaching pipe (hollow 3D structure) - PyTorch")

    try:
        if colors_dict is None:
            colors_dict = {"pipes": ["blue", "cyan", "magenta"]}

        structure = structure.to(device)
        colors = colors.to(device)
        original_sparse_mode = sparse_mode

        width = PyTorchUtils.generate_random_size(
            element_volume_min_ratio, element_volume_max_ratio, structure.shape[0], step, device
        )
        height = PyTorchUtils.generate_random_size(
            element_volume_min_ratio, element_volume_max_ratio, structure.shape[1], step, device
        )
        depth = PyTorchUtils.generate_random_size(
            element_volume_min_ratio, element_volume_max_ratio, structure.shape[2], step, device
        )

        log_info(f"Pipe dimensions: width={width}, height={height}, depth={depth}")

        if not PyTorchUtils.validate_dimensions([width, height, depth], structure.shape):
            log_warning("Pipe dimensions too large for structure, skipping")
            end_section("Pipe attachment skipped")
            return structure, colors

        x_pos = PyTorchUtils.select_random_position(structure.shape[0], width)
        y_pos = PyTorchUtils.select_random_position(structure.shape[1], height)
        z_pos = PyTorchUtils.select_random_position(structure.shape[2], depth)

        log_info(f"Pipe position: x={x_pos}, y={y_pos}, z={z_pos}")

        if axis_selection is None:
            axis_selection = random.randint(0, 1)
        elif axis_selection not in (0, 1):
            raise ValueError("axis_selection must be 0, 1, or None")
        shape_selection = random.randint(0, 1)

        log_info(f"Design parameters: axis_selection={axis_selection}, shape_selection={shape_selection}")

        pipe_color = PyTorchUtils.select_random_color(colors_dict["pipes"])
        log_info(f"Selected color: {pipe_color}")

        if isinstance(pipe_color, str):
            color_value = hash(pipe_color) % 256
        else:
            color_value = pipe_color

        if pipe_complexity == "simple":
            structure, colors = _create_simple_pipe_pytorch(
                structure, colors, x_pos, y_pos, z_pos, width, height, depth,
                color_value, axis_selection, shape_selection, wall_thickness, device
            )
        elif pipe_complexity == "complex":
            structure, colors = _create_complex_pipe_pytorch(
                structure, colors, x_pos, y_pos, z_pos, width, height, depth,
                color_value, wall_thickness, device
            )
        elif pipe_complexity == "curved":
            log_info("Curved pipes not yet implemented, using simple pipe")
            structure, colors = _create_simple_pipe_pytorch(
                structure, colors, x_pos, y_pos, z_pos, width, height, depth,
                color_value, axis_selection, shape_selection, wall_thickness, device
            )
        else:
            raise ValueError(f"Unknown pipe complexity: {pipe_complexity}")

        if original_sparse_mode and SparseTensorHandler.should_use_sparse(structure):
            structure = SparseTensorHandler.to_sparse(structure)
            colors = SparseTensorHandler.to_sparse(colors)
            log_info("Converted to sparse tensor representation")

        if not structure.is_sparse:
            structure = structure.to(torch.int8)

        log_success("Pipe attached successfully")
        end_section()
        return structure, colors

    except Exception as e:
        log_error(f"Error attaching pipe: {str(e)}")
        end_section("Pipe attachment failed")
        raise


def _create_simple_pipe_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    x_pos: int, y_pos: int, z_pos: int,
    width: int, height: int, depth: int,
    color_value: int,
    axis_selection: int,
    shape_selection: int,
    wall_thickness: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple pipe structure using PyTorch tensors.
    """
    # Floor
    structure[x_pos: x_pos + width, y_pos: y_pos + height, z_pos] = 1
    colors[x_pos: x_pos + width, y_pos: y_pos + height, z_pos] = color_value

    # Ceiling
    structure[x_pos: x_pos + width, y_pos: y_pos + height, z_pos + depth - 1] = 1
    colors[x_pos: x_pos + width, y_pos: y_pos + height, z_pos + depth - 1] = color_value

    # Walls
    if shape_selection == 0:
        if axis_selection == 0:
            structure[x_pos, y_pos: y_pos + height, z_pos: z_pos + depth] = 1
            colors[x_pos, y_pos: y_pos + height, z_pos: z_pos + depth] = color_value

            structure[x_pos + width - 1, y_pos: y_pos + height, z_pos: z_pos + depth] = 1
            colors[x_pos + width - 1, y_pos: y_pos + height, z_pos: z_pos + depth] = color_value
        else:
            structure[x_pos: x_pos + width, y_pos, z_pos: z_pos + depth] = 1
            colors[x_pos: x_pos + width, y_pos, z_pos: z_pos + depth] = color_value

            structure[x_pos: x_pos + width, y_pos + height - 1, z_pos: z_pos + depth] = 1
            colors[x_pos: x_pos + width, y_pos + height - 1, z_pos: z_pos + depth] = color_value
    else:
        if axis_selection == 0:
            structure[x_pos, y_pos: y_pos + height, z_pos: z_pos + depth] = 1
            colors[x_pos, y_pos: y_pos + height, z_pos: z_pos + depth] = color_value

            structure[x_pos: x_pos + width, y_pos, z_pos: z_pos + depth] = 1
            colors[x_pos: x_pos + width, y_pos, z_pos: z_pos + depth] = color_value
        else:
            structure[x_pos + width - 1, y_pos: y_pos + height, z_pos: z_pos + depth] = 1
            colors[x_pos + width - 1, y_pos: y_pos + height, z_pos: z_pos + depth] = color_value

            structure[x_pos: x_pos + width, y_pos, z_pos: z_pos + depth] = 1
            colors[x_pos: x_pos + width, y_pos, z_pos: z_pos + depth] = color_value

    return structure, colors


def _create_complex_pipe_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    x_pos: int, y_pos: int, z_pos: int,
    width: int, height: int, depth: int,
    color_value: int,
    wall_thickness: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a complex pipe structure with configurable wall thickness.
    """
    # Outer shell
    structure[x_pos: x_pos + width, y_pos: y_pos + height, z_pos: z_pos + depth] = 1
    colors[x_pos: x_pos + width, y_pos: y_pos + height, z_pos: z_pos + depth] = color_value

    # Hollow interior
    if width > 2 * wall_thickness and height > 2 * wall_thickness and depth > 2 * wall_thickness:
        inner_x_start = x_pos + wall_thickness
        inner_x_end = x_pos + width - wall_thickness
        inner_y_start = y_pos + wall_thickness
        inner_y_end = y_pos + height - wall_thickness
        inner_z_start = z_pos + wall_thickness
        inner_z_end = z_pos + depth - wall_thickness

        structure[inner_x_start: inner_x_end, inner_y_start: inner_y_end, inner_z_start: inner_z_end] = 0
        colors[inner_x_start: inner_x_end, inner_y_start: inner_y_end, inner_z_start: inner_z_end] = 0

    return structure, colors


def create_curved_pipe_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    start_pos: Tuple[int, int, int],
    end_pos: Tuple[int, int, int],
    radius: int,
    color_value: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a curved pipe between two points (future enhancement).
    """
    log_info("Curved pipe generation not yet implemented")
    return structure, colors


def attach_pipes_batch_pytorch(
    structures: torch.Tensor,
    colors: torch.Tensor,
    element_volume_min_ratio: float = 0.1,
    element_volume_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    wall_thickness: int = 1,
    pipe_complexity: str = "simple",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach pipes to multiple structures in batch using PyTorch tensors.
    """
    begin_section("Attaching pipes (batch processing) - PyTorch")

    try:
        batch_size = structures.shape[0]
        log_info(f"Processing batch of {batch_size} structures")

        for i in range(batch_size):
            structures[i], colors[i] = attach_pipe_pytorch(
                structures[i],
                colors[i],
                element_volume_min_ratio,
                element_volume_max_ratio,
                step,
                colors_dict,
                device,
                sparse_mode,
                wall_thickness,
                pipe_complexity,
                verbose=False,
            )

        log_success(f"Batch pipe attachment completed for {batch_size} structures")
        end_section()
        return structures, colors

    except Exception as e:
        log_error(f"Error in batch pipe attachment: {str(e)}")
        end_section("Batch pipe attachment failed")
        raise


def validate_pipe_dimensions_pytorch(
    pipe_size: Tuple[int, int, int],
    structure_shape: Tuple[int, int, int],
    position: Tuple[int, int, int],
    wall_thickness: int = 1,
    device: str = "cpu",
) -> bool:
    """
    Validate that a pipe fits within the structure at the given position.
    """
    pipe_tensor = torch.tensor(pipe_size, device=device)
    structure_tensor = torch.tensor(structure_shape, device=device)
    position_tensor = torch.tensor(position, device=device)

    end_position = position_tensor + pipe_tensor
    bounds_check = torch.all(end_position <= structure_tensor).item()

    min_dims = torch.all(pipe_tensor >= 2 * wall_thickness + 1).item()

    return bounds_check and min_dims

# ======================================================================================
# GRID FUNCTIONS
# ======================================================================================

def attach_grid_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    grid_pattern: str = "regular",  # "regular", "irregular", "random"
    grid_density: float = 0.5,
    column_height_variation: bool = True,
    base_floor: bool = True,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach a grid structure to the structure using PyTorch tensors.
    """
    begin_section("Attaching grid structure - PyTorch")

    try:
        if colors_dict is None:
            colors_dict = {"edges": "red"}

        structure = structure.to(device)
        colors = colors.to(device)
        original_sparse_mode = sparse_mode

        structure_dim = structure.shape[0]
        log_info(f"Creating grid in structure of dimension {structure_dim}")

        if grid_pattern == "regular":
            locations = _generate_regular_grid_pytorch(structure_dim, step, grid_density, device)
        elif grid_pattern == "irregular":
            locations = _generate_irregular_grid_pytorch(structure_dim, step, grid_density, device)
        elif grid_pattern == "random":
            locations = _generate_random_grid_pytorch(structure_dim, step, grid_density, device)
        else:
            raise ValueError(f"Unknown grid pattern: {grid_pattern}")

        log_info(f"Generated {len(locations)} grid positions")

        if column_height_variation:
            heights = torch.randint(
                low=structure_dim * 3 // 4,
                high=structure_dim,
                size=(len(locations),),
                device=device,
            )
        else:
            heights = torch.full((len(locations),), structure_dim * 7 // 8, device=device)

        grid_color = PyTorchUtils.select_random_color(colors_dict["edges"])
        log_info(f"Selected color: {grid_color}")

        if isinstance(grid_color, str):
            color_value = hash(grid_color) % 256
        else:
            color_value = grid_color

        for i, (x, y) in enumerate(locations):
            height = heights[i].item()
            if x < structure_dim and y < structure_dim and height > 0:
                max_height = min(height, structure_dim)
                structure[x, y, 0:max_height] = 1
                colors[x, y, 0:max_height] = color_value
                if verbose:
                    log_info(
                        f"Created column at x={x}, y={y}, height={max_height}",
                        is_last=(i == len(locations) - 1),
                    )

        if base_floor:
            structure[:, :, 0] = 1
            floor_color_value = color_value
            colors[structure[:, :, 0] == 1] = floor_color_value
            log_info("Created base floor")

        if original_sparse_mode and SparseTensorHandler.should_use_sparse(structure):
            structure = SparseTensorHandler.to_sparse(structure)
            colors = SparseTensorHandler.to_sparse(colors)
            log_info("Converted to sparse tensor representation")

        if not structure.is_sparse:
            structure = structure.to(torch.int8)

        log_success(f"Grid created with {len(locations)} columns")
        end_section()
        return structure, colors

    except Exception as e:
        log_error(f"Error attaching grid: {str(e)}")
        end_section("Grid attachment failed")
        raise


def _generate_regular_grid_pytorch(
    structure_dim: int, step: int, density: float, device: str
) -> List[Tuple[int, int]]:
    """Generate regular grid positions — full x×y array of columns at every intersection."""
    locations = []
    for x in range(0, structure_dim, step + 1):
        for y in range(0, structure_dim, step + 1):
            if x < structure_dim and y < structure_dim:
                locations.append((x, y))
    if density < 1.0:
        num_keep = max(1, int(len(locations) * density))
        indices = torch.randperm(len(locations), device=device)[:num_keep]
        locations = [locations[i] for i in indices.cpu().tolist()]
    return locations


def _generate_irregular_grid_pytorch(
    structure_dim: int, step: int, density: float, device: str
) -> List[Tuple[int, int]]:
    """Generate irregular grid positions with some randomness."""
    base_locations = _generate_regular_grid_pytorch(structure_dim, step, 1.0, device)
    irregular_locations = []
    for x, y in base_locations:
        offset_range = max(1, step // 2)
        new_x = max(0, min(structure_dim - 1, x + random.randint(-offset_range, offset_range)))
        new_y = max(0, min(structure_dim - 1, y + random.randint(-offset_range, offset_range)))
        irregular_locations.append((new_x, new_y))
    if density < 1.0:
        num_keep = max(1, int(len(irregular_locations) * density))
        indices = torch.randperm(len(irregular_locations), device=device)[:num_keep]
        irregular_locations = [irregular_locations[i] for i in indices.cpu().tolist()]
    return irregular_locations


def _generate_random_grid_pytorch(
    structure_dim: int, step: int, density: float, device: str
) -> List[Tuple[int, int]]:
    """Generate completely random grid positions."""
    max_positions = (structure_dim // (step + 1)) ** 2
    num_positions = max(1, int(max_positions * density))
    x_positions = torch.randint(0, structure_dim, (num_positions,), device=device)
    y_positions = torch.randint(0, structure_dim, (num_positions,), device=device)
    return list(set((x.item(), y.item()) for x, y in zip(x_positions, y_positions)))


def create_procedural_grid_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    grid_params: Dict[str, Any],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a procedural grid with advanced parameters.
    """
    begin_section("Creating procedural grid - PyTorch")
    try:
        result = attach_grid_pytorch(
            structure,
            colors,
            step=grid_params.get("step", 1),
            colors_dict=grid_params.get("colors_dict", {"edges": "red"}),
            device=device,
            sparse_mode=grid_params.get("sparse_mode", False),
            grid_pattern=grid_params.get("pattern", "regular"),
            grid_density=grid_params.get("density", 0.5),
            column_height_variation=grid_params.get("height_variation", True),
            base_floor=grid_params.get("base_floor", True),
            verbose=grid_params.get("verbose", False),
        )
        log_success("Procedural grid created successfully")
        end_section()
        return result
    except Exception as e:
        log_error(f"Error creating procedural grid: {str(e)}")
        end_section("Procedural grid creation failed")
        raise


def attach_grids_batch_pytorch(
    structures: torch.Tensor,
    colors: torch.Tensor,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    sparse_mode: bool = False,
    grid_pattern: str = "regular",
    grid_density: float = 0.5,
    column_height_variation: bool = True,
    base_floor: bool = True,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attach grids to multiple structures in batch using PyTorch tensors.
    """
    begin_section("Attaching grids (batch processing) - PyTorch")

    try:
        batch_size = structures.shape[0]
        log_info(f"Processing batch of {batch_size} structures")
        for i in range(batch_size):
            structures[i], colors[i] = attach_grid_pytorch(
                structures[i],
                colors[i],
                step,
                colors_dict,
                device,
                sparse_mode,
                grid_pattern,
                grid_density,
                column_height_variation,
                base_floor,
                verbose=False,
            )
        log_success(f"Batch grid attachment completed for {batch_size} structures")
        end_section()
        return structures, colors
    except Exception as e:
        log_error(f"Error in batch grid attachment: {str(e)}")
        end_section("Batch grid attachment failed")
        raise


def validate_grid_parameters_pytorch(
    grid_params: Dict[str, Any],
    structure_shape: Tuple[int, int, int],
    device: str = "cpu",
) -> bool:
    """
    Validate grid parameters for feasibility.
    """
    try:
        step = grid_params.get("step", 1)
        density = grid_params.get("density", 0.5)
        pattern = grid_params.get("pattern", "regular")
        if step <= 0 or step >= min(structure_shape):
            return False
        if density <= 0.0 or density > 1.0:
            return False
        if pattern not in ["regular", "irregular", "random"]:
            return False
        return True
    except Exception:
        return False
