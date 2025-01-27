# The distortion reference file is much more complex than previous reference
# files we've had to convert. In an attempt to determine what information
# is required to gather from our Product Owner/external Subject Matter Experts,
# I'll attempt to break down the following code into its individual components,
# most derived from pysiaf

from astropy.modeling.models import Polynomial2D, Mapping, Shift, SIP
import astropy.units as u
from astropy.io import fits
from jwst.datamodels import DistortionModel
import numpy as np

from stdatamodels import util


def get_distortion_coeffs(degree, filter_info):
    """Do this with the grism file header input instead"""
    a_coeffs = {}
    b_coeffs = {}

    for key in filter_info:
        if key[0:2] == "A_" or key[0:2] == "B_":
            if "ORDER" in key:
                continue
            split_key = key.split("_")
            new_key = "c{}_{}".format(split_key[1], split_key[2])
            if split_key[0] == "A":
                a_coeffs[new_key] = filter_info[key]
            elif split_key[0] == "B":
                b_coeffs[new_key] = filter_info[key]

    return a_coeffs, b_coeffs


def get_SIP_Model():
    """Constructs the SIP Distortion Astropy Model using the coefficients file
    provided by Russell Ryan
    Paramters
    ---------
    Returns
    -------
    SIP_model : astropy.modeling.models.SIP
        An Astropy SIP model encoding instrument distortion
    """
    hduIRHeader = fits.open("hst_wfc3_ir_fov.fits")['IR'].header
    degree = hduIRHeader['AP_ORDER']
    if degree != hduIRHeader['BP_ORDER']:
        raise ValueError("AP and BP orders not equal!")

    import re

    # Create dictionaries of distortion coefficients
    coeffs = {
        'A': {
            'pattern': re.compile("A_\d_\d"),  # noqa: W605
            'matching_coeffs': dict()
        },
        'B': {
            'pattern': re.compile("B_\d_\d"),  # noqa: W605
            'matching_coeffs': dict()
        },
        'AP': {
            'pattern': re.compile("AP_\d_\d"),  # noqa: W605
            'matching_coeffs': dict()
        },
        'BP': {
            'pattern': re.compile("AB_\d_\d"),  # noqa: W605
            'matching_coeffs': dict()
        }
    }

    for key in hduIRHeader:
        for coeff in coeffs.values():
            if coeff['pattern'].match(key):
                coeff['matching_coeffs'][key] = hduIRHeader[key]

    SIP_model = SIP(crpix=[
                        hduIRHeader['CRPIX1'],
                        hduIRHeader['CRPIX2']
                    ],
                    a_order=hduIRHeader['A_ORDER'],
                    a_coeff=coeffs['A']['matching_coeffs'],
                    b_order=hduIRHeader['B_ORDER'],
                    b_coeff=coeffs['B']['matching_coeffs'],
                    ap_order=hduIRHeader['AP_ORDER'],
                    ap_coeff=coeffs['AP']['matching_coeffs'],
                    bp_order=hduIRHeader['BP_ORDER'],
                    bp_coeff=coeffs['BP']['matching_coeffs']

                    )
    return SIP_model


def v2v3_model(from_sys, to_sys, par, angle):
    """
    Creates an astropy.modeling.Model object
    for the undistorted ("ideal") to V2V3 coordinate translation
    """
    if from_sys != 'v2v3' and to_sys != 'v2v3':
        raise ValueError("Only transformation either to or from V2V3 are supported")

    # Cast the transform functions as 1st order polynomials
    xc = {}
    yc = {}
    if to_sys == 'v2v3':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = np.sin(angle)
        yc['c1_0'] = (0.-par) * np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    if from_sys == 'v2v3':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = par * (0. - np.sin(angle))
        yc['c1_0'] = np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    # 0,0 coeff should never be used.
    xc['c0_0'] = 0
    yc['c0_0'] = 0

    xmodel = Polynomial2D(1, **xc)
    ymodel = Polynomial2D(1, **yc)

    return xmodel, ymodel


def create_distortion(outname,
                      sci_pupil,
                      sci_subarr,
                      sci_exptype,
                      detector="None",
                      history_entry="Astrogrism HST Distortion",
                      save_to_asdf=False):
    """
    https://github.com/spacetelescope/nircam_calib/blob/master/nircam_calib/reffile_creation/pipeline/distortion/nircam_distortion_reffiles_from_pysiaf.py#L37
    Create an asdf reference file with all distortion components for the NIRCam imager.
    NOTE: The IDT has not provided any distortion information. The files are constructed
    using ISIM transformations provided/(computed?) by the TEL team which they use to
    create the SIAF file.
    These reference files should be replaced when/if the IDT provides us with distortion.
    Parameters
    ----------
    detector : str
        NRCB1, NRCB2, NRCB3, NRCB4, NRCB5, NRCA1, NRCA2, NRCA3, NRCA4, NRCA5
    aperture : str
        Name of the aperture/subarray. (e.g. FULL, SUB160, SUB320, SUB640, GRISM_F322W2)
    outname : str
        Name of output file.
    Examples
    --------
    """
    # Download WFC3 Image Distortion File
    from astropy.utils.data import download_file

    # Raw FLT Grism image file with encoded SIP Distortion Coeffients
    fn = download_file('https://github.com/npirzkal/aXe_WFC3_Cookbook/raw/main/cookbook_data/G141/ib6o23rsq_flt.fits', cache=True)  # noqa: E501
    grism_image_hdulist = fits.open(fn)
    distortion_info = grism_image_hdulist['SCI'].header

    degree = 4  # WFC3 Distortion is fourth degree

    # From Bryan Hilbert:
    #   The parity term is just an indicator of the relationship between the detector y axis
    #   and the “science” y axis. A parity of -1 means that the y axes of the two systems run
    #   in opposite directions... A value of 1 indicates no flip.
    # From Colin Cox:
    #   ... for WFC3 it is always -1 so maybe people gave up mentioning it.
    parity = -1

    # *****************************************************
    # "Forward' transformations. science --> ideal --> V2V3

    # With SIP coefficients from the FITS header
    xcoeffs, ycoeffs = get_distortion_coeffs(degree, distortion_info)

    # Get info for ideal -> v2v3 or v2v3 -> ideal model
    idl2v2v3x, idl2v2v3y = v2v3_model('ideal',
                                      'v2v3',
                                      parity,
                                      np.radians(distortion_info["IDCTHETA"]))

    # Now create a compound model for each with the appropriate inverse
    # Inverse polynomials were removed in favor of using GWCS' numerical inverse capabilities
    sci2idl = get_SIP_Model()

    idl2v2v3 = Mapping([0, 1, 0, 1]) | idl2v2v3x & idl2v2v3y

    # Now string the models together to make a single transformation

    # We also need
    # to account for the difference of 1 between the SIAF
    # coordinate values (indexed to 1) and python (indexed to 0).
    # Nadia said that this shift should be present in the
    # distortion reference file.

    core_model = sci2idl | idl2v2v3

    # Now add in the shifts to create the full model
    # including the shift to go from 0-indexed python coords to
    # 1-indexed

    # Find the distance between (0,0) and the reference location
    xshift = Shift(distortion_info['IDCXREF'])
    yshift = Shift(distortion_info['IDCYREF'])

    # Finally, we need to shift by the v2,v3 value of the reference
    # location in order to get to absolute v2,v3 coordinates
    v2shift = Shift(distortion_info['IDCV2REF'])
    v3shift = Shift(distortion_info['IDCV3REF'])

    # SIAF coords
    index_shift = Shift(1)
    model = index_shift & index_shift | xshift & yshift | core_model | v2shift & v3shift

    # Since the inverse of all model components are now defined,
    # the total model inverse is also defined automatically

    # Save using the DistortionModel datamodel
    d = DistortionModel(model=model, input_units=u.pix,
                        output_units=u.arcsec)

    if save_to_asdf:
        # Populate metadata

        # Keyword values in science data to which this file should
        # be applied
        p_pupil = ''
        for p in sci_pupil:
            p_pupil = p_pupil + p + '|'

        p_subarr = ''
        for p in sci_subarr:
            p_subarr = p_subarr + p + '|'

        p_exptype = ''
        for p in sci_exptype:
            p_exptype = p_exptype + p + '|'

        d.meta.instrument.p_pupil = p_pupil
        d.meta.subarray.p_subarray = p_subarr
        d.meta.exposure.p_exptype = p_exptype

        # metadata describing the reference file itself
        d.meta.title = "WFC3 Distortion"
        d.meta.instrument.name = "WFC3"
        d.meta.instrument.module = detector[-2]

        numdet = detector[-1]
        d.meta.instrument.channel = "LONG" if numdet == '5' else "SHORT"
        # In the reference file headers, we need to switch NRCA5 to
        # NRCALONG, and same for module B.
        d.meta.instrument.detector = (detector[0:4] + 'LONG') if numdet == 5 else detector

        d.meta.telescope = 'HST'
        d.meta.subarray.name = 'FULL'
        d.meta.pedigree = 'GROUND'
        d.meta.reftype = 'DISTORTION'
        d.meta.author = 'D. Nguyen'
        d.meta.litref = "https://github.com/spacetelescope/jwreftools"
        d.meta.description = "Distortion model from SIAF coefficients in pysiaf version 0.6.1"
        d.meta.exp_type = sci_exptype
        d.meta.useafter = "2014-10-01T00:00:00"

        # To be ready for the future where we will have filter-dependent solutions
        d.meta.instrument.filter = 'N/A'

        # Create initial HISTORY ENTRY
        sdict = {'name': 'nircam_distortion_reffiles_from_pysiaf.py',
                 'author': 'B.Hilbert',
                 'homepage': 'https://github.com/spacetelescope/jwreftools',
                 'version': '0.8'}

        entry = util.create_history_entry(history_entry, software=sdict)
        d.history = [entry]

        d.save(outname)
        print("Output saved to {}".format(outname))
    else:
        return d
