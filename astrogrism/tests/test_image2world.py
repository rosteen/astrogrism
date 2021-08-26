import pathlib

from astrogrism import GrismObs
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.wcs import WCS
import numpy as np
import pytest

# TODO: Switch to importlib
test_dir = pathlib.Path(__file__).parent.absolute()
testdata = [
    download_file('https://github.com/npirzkal/aXe_WFC3_Cookbook/raw/main/cookbook_data/G141/ib6o23rsq_flt.fits'), # noqa
    str(test_dir / 'data' / 'IRG102_icwz15e7q_flt.fits')
]


@pytest.mark.parametrize('grism_image', testdata)
def test_wfc3_astropywcs(grism_image):
    """
    Tests the Astrogrism Detector > World Transform against
    the built in Astropy WCS model
    """
    # Define Pixel Grid
    xx = np.arange(0, 1014, 20)
    yy = np.arange(0, 1014, 20)
    # Download file and initialize wcs
    grism_image_hdulist = fits.open(grism_image)
    astropy_wcs = WCS(grism_image_hdulist['SCI'].header)
    # Convert Pixel Grid to Sky Grid
    astropy_coords = astropy_wcs.pixel_to_world(xx, yy)
    # Extract sky coordinates
    astropywcs_ra = astropy_coords.ra.value
    astropywcs_dec = astropy_coords.dec.value

    # Init GrismObs
    grismobs = GrismObs(grism_image)
    # Retrieve Transform
    image2world = grismobs.geometric_transforms.get_transform('detector',
                                                              'world')
    # Calculate sky coordinates on grid
    astrogrism_ra, astrogrism_dec = image2world(xx, yy, 0, 0)[:2]

    # Compare results
    np.testing.assert_allclose(astrogrism_ra, astropywcs_ra, atol=5e-02)
    np.testing.assert_allclose(astrogrism_dec, astropywcs_dec, atol=5e-02)
