import zodiax as zdx
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu


# A simple model of a Euclid-like telescope
class AngularOpticalSystem(dl.AngularOpticalSystem):
    filters: dict

    def __init__(self, *args, filters=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters


def euclid_aperture(
    npix,
    diameter,
    oversample=5,
    obscuration_ratio=0.2,
    spider_width_m=0.05,
    spider_angles_deg=[0, 120, 240],
):
    # Derived quantities
    m2_diam = obscuration_ratio * diameter
    spider_shift = m2_diam / 2 - spider_width_m / 2

    # Primary and secondary mirror
    ap_coords = dlu.pixel_coords(npix * oversample, diameter)
    primary = dlu.circle(ap_coords, diameter / 2)
    secondary = dlu.circle(ap_coords, m2_diam / 2, invert=True)

    # Spider vanes
    bars = []
    for angle in spider_angles_deg:
        sp_coord = dlu.rotate_coords(ap_coords, np.radians(angle + 30))
        sp_coord = dlu.translate_coords(sp_coord, np.array([spider_shift, 0]))
        sp_coord = dlu.translate_coords(sp_coord, np.array([0, diameter / 2]))
        bar = dlu.rectangle(
            sp_coord, width=spider_width_m, height=diameter, invert=True
        )
        bars.append(bar)

    # Combine and downsample back to npix
    aperture = dlu.combine([primary, secondary] + bars, oversample)

    return aperture


def zernike_basis(npix, diameter, z_inds):
    coords = dlu.pixel_coords(npix, diameter)
    zern_basis = dlu.zernike_basis(z_inds, coords, diameter)
    return zern_basis


def build_optics(npix=256, diameter=1.2, filters=None):

    aperture = euclid_aperture(
        npix=npix,
        diameter=diameter,
        oversample=5,
        obscuration_ratio=0.2,
        spider_width_m=0.05,
        spider_angles_deg=[0, 120, 240],
    )

    zern_basis = zernike_basis(
        npix=npix, diameter=diameter, z_inds=np.arange(3, 10) + 1
    )

    # Construct the optical system
    return AngularOpticalSystem(
        wf_npixels=npix,
        diameter=diameter,
        layers=[
            ("aperture", dl.TransmissiveLayer(aperture, normalise=True)),
            ("wfe", dl.BasisLayer(zern_basis)),
        ],
        psf_npixels=32,
        psf_pixel_scale=75e-3,
        filters=filters,
    )
