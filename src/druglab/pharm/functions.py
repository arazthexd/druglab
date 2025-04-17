from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Union, Generator, Any

import numpy as np

def geom_direction(variables: Dict[Union[int,str], Any],
                   output_keys: List[str],
                   extra: Dict[str, Any],
                   input_keys: List[str] = None) -> None:
    if input_keys is None:
        input_keys = [0, 1]
    x1 = variables[input_keys[0]]
    x2 = variables[input_keys[1]]

    x12 = x2 - x1
    variables[output_keys[0]] = x12 / np.linalg.norm(x12)
    return

def geom_mean(variables: Dict[Union[int,str], Any],
              output_keys: List[str],
              extra: Dict[str, Any],
              input_keys: List[str] = None) -> None:
    if input_keys is None:
        input_keys = [key for key in variables.keys() if isinstance(key, int)]

    s = sum([variables[key] for key in input_keys])
    s = s / len(input_keys)
    variables[output_keys[0]] = s
    return

def geom_minus(variables: Dict[Union[int,str], Any],
               output_keys: List[str],
               extra: Dict[str, Any],
               input_keys: List[str] = None) -> None:
    if input_keys is None:
        input_keys = [0]

    variables[output_keys[0]] = -variables[input_keys[0]]
    return

def geom_perpendicular3to1(variables: Dict[Union[int,str], Any],
                           output_keys: List[str],
                           extra: Dict[str, Any],
                           input_keys: List[str] = None) -> None:
    
    if input_keys is None:
        input_keys = [0, 1, 2]
    xc = variables[input_keys[0]]
    x1 = variables[input_keys[1]]
    x2 = variables[input_keys[2]]

    v1 = xc - x1
    v2 = xc - x2
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    h = np.cross(v1, v2)
    h = h / np.linalg.norm(h)

    variables[output_keys[0]] = h
    return

def geom_angle_2to1_point(variables: Dict[Union[int,str], Any],
                          output_keys: List[str],
                          extra: Dict[str, Any],
                          input_keys: List[str] = None) -> None:
    
    angle = extra.get("angle", 109.5)
    dist = extra.get("dist", 1.0)

    if input_keys is None:
        input_keys = [0, 1]
    xc = variables[input_keys[0]]
    xo = variables[input_keys[1]]

    v = xo - xc
    v = v / np.linalg.norm(v)

    # find two perpendicular vectors.
    if np.allclose(v, [0, 0, 1]) or np.allclose(v, [0, 0, -1]):
        p1 = np.array([1, 0, 0])
    else:
        p1 = np.cross(v, [0, 0, 1])
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(v, p1)
    p2 = p2 / np.linalg.norm(p2)

    # find radius of circle + how far to go in dir of xc-xo
    angle = np.deg2rad(angle)
    radius = dist * np.sin(angle)
    forward = dist * np.cos(angle)

    # sample random angle
    theta = np.random.uniform(0, 2 * np.pi)
    
    # get delta of a point with respect to xc
    delta = forward * v + radius * (p1 * np.cos(theta) + p2 * np.sin(theta))

    h1 = xc + delta

    variables[output_keys[0]] = h1
    return

def geom_extended_plane_3to2(variables: Dict[Union[int,str], Any],
                             output_keys: List[str],
                             extra: Dict[str, Any],
                             input_keys: List[str] = None) -> None:
    theta = extra.get("theta", 60.0)

    if input_keys is None:
        input_keys = [0, 1, 2]
    xc: np.ndarray = variables[input_keys[0]]
    xk: np.ndarray = variables[input_keys[1]]
    xr: np.ndarray = variables[input_keys[2]]
    
    v1 = xc - xk
    v1 = v1 / np.linalg.norm(v1)

    v2 = xk - xr 
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    theta = np.deg2rad(theta)
    h1 = v1 * np.cos(theta) + v2 * np.cos(theta)
    h2 = v1 * np.cos(theta) - v2 * np.cos(theta)
    h1 = h1 / np.linalg.norm(h1)
    h2 = h2 / np.linalg.norm(h2)

    variables[output_keys[0]], variables[output_keys[1]] = h1, h2
    return

def geom_plane_3to1(variables: Dict[Union[int,str], Any],
                    output_keys: List[str],
                    extra: Dict[str, Any],
                    input_keys: List[str] = None) -> None:
    
    if input_keys is None:
        input_keys = [0, 1, 2]
    xc = variables[input_keys[0]]
    x1 = variables[input_keys[1]]
    x2 = variables[input_keys[2]]

    v1 = xc - x1
    v2 = xc - x2
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    h = v1 + v2
    h = h / np.linalg.norm(h)

    variables[output_keys[0]] = h
    return

def geom_tetrahedral_3to2(variables: Dict[Union[int,str], Any],
                          output_keys: List[str],
                          extra: Dict[str, Any],
                          input_keys: List[str] = None) -> None:
    theta = extra.get("theta", 120.0)

    if input_keys is None:
        input_keys = [0, 1, 2]
    xc = variables[input_keys[0]]  # point A (center)
    x1 = variables[input_keys[1]]  # point B
    x2 = variables[input_keys[2]]  # point C
    
    # Get vector BA (rotation axis)
    ba = x1 - xc
    ba_unit = ba / np.linalg.norm(ba)
    
    # Get vector to rotate (AC)
    ac = x2 - xc
    
    # Rodrigues rotation formula for 120 and -120 degrees
    angle_120 = np.deg2rad(theta)
    cos_theta = np.cos(angle_120)
    sin_theta = np.sin(angle_120)
    
    # First rotation (+120 degrees)
    ac1 = (ac * cos_theta + 
           np.cross(ba_unit, ac) * sin_theta + 
           ba_unit * np.dot(ba_unit, ac) * (1 - cos_theta))
    
    # Second rotation (-120 degrees)
    ac2 = (ac * cos_theta - 
           np.cross(ba_unit, ac) * sin_theta + 
           ba_unit * np.dot(ba_unit, ac) * (1 - cos_theta))
    
    # Scale vectors to desired distances
    ac1_unit = ac1 / np.linalg.norm(ac1)
    ac2_unit = ac2 / np.linalg.norm(ac2)
    
    variables[output_keys[0]], variables[output_keys[1]] = ac1_unit, ac2_unit
    return

import numpy as np
from typing import Any, Dict, List, Union

def geom_tetrahedral_3to2_new(variables: Dict[Union[int, str], Any],
                              output_keys: List[str],
                              extra: Dict[str, Any],
                              input_keys: List[Union[int, str]] = None) -> None:
    """
    Given a central atom at x_c and two bonded atoms at x1 and x2,
    compute two lone pair directions (l1 and l2) that obey the ideal
    sp³ (tetrahedral) condition that each lone pair makes an angle ~109.47° with each bond.
    
    In a perfect tetrahedral arrangement the dot product between any two directions is -1/3.
    Here we enforce b1·l = b2·l = -1/3, where b1 and b2 are the normalized bond vectors.
    
    The ansatz is:
        l = A*(b1 + b2) ± B*(b1 x b2)
    with A = -1/(3*(1+d)) (d = b1·b2) and B chosen (by normalization) as:
        B = sqrt( (1 - 2/(9*(1+d))) / (1-d²) )
    
    This gives two lone-pair directions even if the bond angle deviates from 109.5°.
    """
    # Default input keys if not provided
    if input_keys is None:
        input_keys = [0, 1, 2]
        
    # Set up positions
    xc = np.array(variables[input_keys[0]], dtype=float)  # central atom (e.g. oxygen)
    x1 = np.array(variables[input_keys[1]], dtype=float)  # bonded atom 1 (e.g. hydrogen)
    x2 = np.array(variables[input_keys[2]], dtype=float)  # bonded atom 2

    # Compute normalized bond vectors
    b1 = x1 - xc
    b1 /= np.linalg.norm(b1)
    b2 = x2 - xc
    b2 /= np.linalg.norm(b2)
    
    # Compute dot product between bond vectors
    d = np.dot(b1, b2)

    # In the degenerate case of nearly colinear bonds, choose an arbitrary perpendicular for lone pairs.
    if np.abs(1.0 - d) < 1e-6:
        # Bonds are nearly parallel; pick a perpendicular direction
        cp = np.cross(b1, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(cp) < 1e-6:
            cp = np.cross(b1, np.array([0.0, 1.0, 0.0]))
        cp /= np.linalg.norm(cp)
        l1 = -b1  # simply opposite in direction
        l2 = -b2
    else:
        # Determine A coefficient so that b1 · l = A*(1+d) = -1/3
        A = -1.0 / (3.0 * (1.0 + d))
    
        # Compute the cross product; this vector is perpendicular to the plane of b1 and b2.
        cp = np.cross(b1, b2)
        norm_cp = np.linalg.norm(cp)
        if norm_cp < 1e-8:
            # Fallback in the unlikely event b1 and b2 are colinear (should be caught above)
            cp = np.cross(b1, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(cp) < 1e-8:
                cp = np.cross(b1, np.array([0.0, 1.0, 0.0]))
            norm_cp = np.linalg.norm(cp)
        cp_unit = cp / norm_cp

        # Now, require that l be unit length. Since b1+b2 and b1×b2 are orthogonal:
        #   ||l||² = A²||b1+b2||² + B²||b1×b2||² = 1.
        # Note: ||b1+b2||² = 2(1+d) and ||b1×b2||² = 1-d².
        # Solve for B:
        denom = 1.0 - d**2
        if denom <= 0:
            B = 0.0
        else:
            B_sq = (1.0 - 2.0/(9.0*(1.0+d))) / denom
            B = np.sqrt(B_sq) if B_sq > 0.0 else 0.0

        # The two lone pair directions, ensuring that:
        #   b1 · l_i = A*(1+d) = -1/3   for i = 1,2.
        l1 = A*(b1 + b2) + B*cp_unit
        l2 = A*(b1 + b2) - B*cp_unit

        # Normalize (to remove any numerical drift)
        l1 /= np.linalg.norm(l1)
        l2 /= np.linalg.norm(l2)
    
    # Store the computed lone pair directions into the output keys.
    variables[output_keys[0]] = l1
    variables[output_keys[1]] = l2
    return
