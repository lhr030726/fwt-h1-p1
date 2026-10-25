"""Geographic wind to image directions using audited per-event georeferencing.

The local projection Jacobian is estimated with symmetric one-metre geodesics.
This accounts for Albers convergence AND its unequal directional scales. The
resulting image direction is normalized to retain the supplied speed magnitude
in its stored units. This does not verify those units or predict displacement.
"""
from functools import lru_cache
import numpy as np


def validate_georeference(value, shape, pixel_size_m):
    if not isinstance(value, dict) or not isinstance(value.get('crs'), str):
        raise ValueError('Geographic wind requires explicit per-sample georeference with CRS')
    affine = np.asarray(value.get('affine_transform'), dtype=np.float64)
    if affine.shape != (6,) or not np.isfinite(affine).all():
        raise ValueError('georeference requires six finite affine_transform values')
    # Terrain and distance currently assume a square, north-up metre grid.
    if not np.allclose(affine[[1, 2, 4, 5]], [pixel_size_m, 0, 0, -pixel_size_m], rtol=0, atol=1e-8):
        raise ValueError('Only north-up square grids matching pixel_size_m are supported')
    if list(value.get('shape', [])) != list(shape):
        raise ValueError('georeference shape differs from the input array')
    return tuple(affine.tolist()), value['crs']


@lru_cache(maxsize=32)
def _image_basis(affine, crs_text, shape):
    from pyproj import CRS, Transformer

    crs = CRS.from_user_input(crs_text)
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-12 for a in crs.axis_info):
        raise ValueError('georeference CRS must be projected with metre axes')
    geographic = crs.geodetic_crs
    inverse = Transformer.from_crs(crs, geographic, always_xy=True)
    forward = Transformer.from_crs(geographic, crs, always_xy=True)
    geod = geographic.get_geod()
    rows, cols = np.indices(shape, dtype=np.float64)
    # Evaluate the local vector basis at pixel centres. Integer affine nodes
    # used to reproduce a producer's nearest-weather-cell partition describe a
    # separate sampling operation and do not change this Jacobian convention.
    x = affine[0] + (cols + .5) * affine[1] + (rows + .5) * affine[2]
    y = affine[3] + (cols + .5) * affine[4] + (rows + .5) * affine[5]
    lon, lat = inverse.transform(x, y, errcheck=True)
    matrix = np.array([[affine[1], affine[2]], [affine[4], affine[5]]])
    inverse_affine = np.linalg.inv(matrix)
    basis = []
    for bearing in (90., 0.):  # Geographic east, then north; azimuth is clockwise from north.
        endpoints = []
        for direction in (bearing, bearing + 180.):
            lo, la, _ = geod.fwd(lon, lat, np.full(shape, direction), np.ones(shape))
            ex, ey = forward.transform(lo, la, errcheck=True)
            endpoints.append(np.stack([ex, ey]))
        derivative = (endpoints[0] - endpoints[1]) / 2.
        basis.append(np.einsum('ij,jhw->ihw', inverse_affine, derivative))
    result = np.stack(basis, axis=1)  # image component, geographic component, H, W
    if not np.isfinite(result).all() or np.any(np.abs(np.linalg.det(result.transpose(2, 3, 0, 1))) < 1e-12):
        raise ValueError('Invalid local geographic-to-image basis')
    result.setflags(write=False)
    return result


def geographic_wind(speed, angle, georeference, pixel_size_m):
    """Angle hypothesis: geographic east origin, counterclockwise, towards.

    This routine verifies coordinate conversion, not the original producer's
    angle convention. The latter must be recorded separately in the data audit.
    """
    affine, crs = validate_georeference(georeference, speed.shape, pixel_size_m)
    basis = _image_basis(affine, crs, tuple(speed.shape))
    direction = np.einsum('ijhw,jhw->ihw', basis, np.stack([np.cos(angle), np.sin(angle)]))
    length = np.linalg.norm(direction, axis=0)
    if not np.isfinite(length).all() or np.any(length < 1e-12):
        raise ValueError('Degenerate projected wind direction')
    vector = direction / length[None] * speed[None]
    return vector[0].astype(np.float32), vector[1].astype(np.float32)
