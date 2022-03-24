from math import floor
from os import environ
from pathlib import Path
from shutil import copy
from tarfile import open as tar_open
from tempfile import gettempdir
from warnings import warn

from astropy.utils.data import download_file
from astropy.io.fits import ImageHDU
import astropy.units as u
from synphot import Observation
import numpy as np
import progressbar

from astrogrism import GrismObs


def download_stsynphot_files(download_path):

    # Prepare environment required for stsynphot to be imported
    environ['PYSYN_CDBS'] = str(download_path / 'grp' / 'redcat' / 'trds')

    # Download HST Instrument Data Archive
    download_path.mkdir(parents=True, exist_ok=True)
    hst_data_files_archive = Path(download_file('https://ssb.stsci.edu/'
                                                'trds/tarfiles/synphot1.tar.gz', cache=True))
    with tar_open(hst_data_files_archive) as tar:
        # Only extract the files that are missing
        for file in tar:
            if not (download_path / Path(file.name)).exists():
                tar.extract(file, path=download_path)
    # Download Vega CALSPEC Reference Atlas
    vega_reference_atlas_path = download_path / 'grp/redcat/trds/calspec/alpha_lyr_stis_010.fits'
    # Check if it exists first before trying to download it
    if not vega_reference_atlas_path.exists():
        vega_reference_atlas_path.parent.mkdir(parents=True, exist_ok=True)
        archive_url = ('https://archive.stsci.edu/hlsps/reference-atlases/cdbs/'
                       'current_calspec/alpha_lyr_stis_010.fits')
        temp_download = Path(download_file(archive_url, cache=True))
        copy(str(temp_download), str(vega_reference_atlas_path))


def generate_synthetic_spectrum(grism, detector=None, temp_path=gettempdir(), verbose=False):
    """
    Initializes and uses STSynphot to generate a Vega spectrum within the bandpass of a given grism

    Parameters
    ----------
    grism : str
        String representation of one of the four supported HST Grisms
        Valid grisms: G141, G102, G280, G800L

    detector : int
        For detectors with multiple chips, specifies which chip to simulate
        Only useful for G280 and G800L Grisms

    temp_path : str
        Path to download necessary files for STSynphot. Fallsback to Python's
        default temporary folder location

    """
    if detector not in (1, 2, None):
        raise ValueError("Invalid detector argument. Please choose 1 or 2")

    SIM_DATA_DIR = Path(temp_path) / "astrogrism_simulation_files"

    download_stsynphot_files(SIM_DATA_DIR)

    # Now that we have all our reference files, we can import stsynphot
    # (This is why it's not a top-line import)
    from stsynphot import Vega, band
    if grism == 'G141':
        if detector and verbose:
            warn("WFC3's G141 grism does not have multiple detectors. Ignoring detector argument",
                 RuntimeWarning)
        bandpass = band('wfc3,ir,g141')
    elif grism == 'G102':
        if detector and verbose:
            warn("WFC3's G102 grism does not have multiple detectors. Ignoring detector argument",
                 RuntimeWarning)
        bandpass = band('wfc3,ir,g102')
    elif grism == 'G280':
        bandpass = band(f'wfc3,uvis{detector},g280')
    elif grism == 'G800L':
        bandpass = band(f'acs,wfc{detector},g800l')
    else:
        raise ValueError(f"Unrecognized grism: {grism}. Valid grisms: G141, G102, G280, G800L")

    spectrum = Observation(Vega, bandpass, binset=bandpass.binset).to_spectrum1d()

    # Find the first value with a non-zero "flux"
    for i in range(len(spectrum.flux.value)):
        value = spectrum.flux.value[i]
        if value != 0.0:
            min_slice = i
            break

    # Find the last value with a non-zero "flux"
    for i in reversed(range(len(spectrum.flux.value))):
        value = spectrum.flux.value[i]
        if value != 0.0:
            max_slice = i
            break
    return spectrum[min_slice:max_slice]


def disperse_spectrum_on_image(grism_file, wide_field_image, spectrum, detector=None):
    """
    Disperses a given spectrum onto an astronomical image
    along the spectral trace of a specified HST grism

    Parameters
    ----------
    grism_file : str or astropy.io.fits.HDUList
        Grism image (or filepath to one) to construct an Astrogrism GrismObs class
        TBF: Should be replaced with a Grism identifier only in the future

    wide_field_image : numpy.ndarray or astropy.io.fits.ImageHDU
        The image or array to disperse the given spectrum atop

    spectrum : specutils.Spectrum1D
        The spectrum to diperse the on the supplied image

    detector : int
        For detectors with multiple chips, specifies which chip to simulate
        Only useful for G280 and G800L Grisms
    """
    grismobs = GrismObs(grism_file)
    if detector:
        image2grism = grismobs.geometric_transforms[f'CCD{detector}'].get_transform('detector',
                                                                                    'grism_detector'
                                                                                    )
    else:
        image2grism = grismobs.geometric_transforms.get_transform('detector', 'grism_detector')

    if type(wide_field_image) is ImageHDU:
        data = wide_field_image.data
    else:
        data = wide_field_image*u.Jy

    # A stupid way of getting all zeroes while preserving units if a Quantity
    simulated_data = data-data

    print(spectrum)
    normalized_spec_flux = spectrum.flux / spectrum.flux.sum()

    x_flat, y_flat = np.meshgrid(np.arange(0, data.shape[0]), np.arange(0, data.shape[1]))
    x_flat = x_flat.flatten()
    y_flat = y_flat.flatten()
    data_flux = data.flatten()

    # Start progressbar
    bar = progressbar.ProgressBar(maxval=spectrum.wavelength.shape[0],
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ',
                                           progressbar.Percentage()
                                           ]
                                  )
    bar.start()

    # For each pixel in the science image, we need to disperse it's spectrum
    for i in range(0, spectrum.wavelength.shape[0]):
        bar.update(i)
        # Get the flux of the science pixel
        # We'll need to scale the spectrum to this brightness later
        #data_flux = data[x_pixel][y_pixel]

        # For each Wavelength in the spectrum,
        # calculate where, in pixels, that wavelength would fall on the detector
        dispersed_coords = image2grism(x_flat, y_flat, spectrum.wavelength[i], 1)
        # TBF: Substitute floor with proper subpixel drizzling
        dispersed_x = np.floor(dispersed_coords[0])
        dispersed_y = np.floor(dispersed_coords[1])

        # Find which dispersed coordinates actually lie on the detector
        good_coords = np.where((dispersed_x > 0) & (dispersed_x < data.shape[0]) &
                               (dispersed_y > 0) & (dispersed_y < data.shape[1]))

        s_x = dispersed_x[good_coords].astype(int)
        s_y = dispersed_y[good_coords].astype(int)
        #print(f"s_x: {s_x}")
        #print(f"x_flat: {x_flat[good_coords]}")

        temp_flux = data*normalized_spec_flux[i]

        simulated_data[s_x, s_y] += temp_flux[x_flat[good_coords], y_flat[good_coords]]

    bar.finish()
    return simulated_data


def simulate_grism(grism, wide_field_image, detector=None, spectrum=None):
    """
    Simulates a grism observation on an astronomical image given a specific grism

    Parameters
    ----------
    grism : str
        String representation of one of the four supported HST Grisms
        Valid grisms: G141, G102, G280, G800L

    wide_field_image : numpy.ndarray or astropy.io.fits.ImageHDU
        The image or array to disperse the given spectrum atop

    detector : int (optional)
        For detectors with multiple chips, specifies which chip to simulate
        Only required for G280 and G800L Grisms

    spectrum: specutils.Spectrum1D (optional)
        The spectrum to disperse onto the image. If none is provided, a Vega synthetic spectrum
        generated by STSynphot will be used.
    """
    test_data_dir = Path(__file__).parent.parent.absolute() / 'tests' / 'data'
    if grism == 'G102':
        grism_file = test_data_dir / 'IRG102_icwz15e7q_flt.fits'
    elif grism == 'G141':
        grism_file = test_data_dir / 'IRG141_ib6o23rsq_flt.fits'
    elif grism == 'G280':
        grism_file = test_data_dir / 'uvis_test_file.fits'
    elif grism == 'G800L':
        grism_file = test_data_dir / 'acs_test_file.fits'
    else:
        raise ValueError(f"Unrecognized grism: {grism}. Valid grisms: G141, G102, G280, G800L")

    if not spectrum:
        spectrum = generate_synthetic_spectrum(grism, detector)
    return disperse_spectrum_on_image(str(grism_file), wide_field_image, spectrum, detector)
