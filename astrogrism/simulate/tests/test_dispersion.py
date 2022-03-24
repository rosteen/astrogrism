import numpy as np
import pytest
from astrogrism.simulate import generate_synthetic_spectrum, simulate_grism

grism_detectors = [
    ('G141', None),
    ('G102', None),
    ('G280', 1),
    ('G280', 2),
    ('G800L', 1),
    ('G800L', 2)
]


@pytest.mark.parametrize("grism,detector", grism_detectors)
def test_sim_sum(grism, detector):
    # Generate spectrum and sum the total flux through the bandpass
    spectrum = generate_synthetic_spectrum(grism, detector)
    expected_sum = np.sum(spectrum.flux.value)

    # Simulate Grism Observation
    # Create an arbitrarily large field with a single point in the center of flux 1
    data = np.zeros((4000, 500))
    data[2000, 250] = 1
    # Simulate grism observation and sum the dispersed flux
    simulation = simulate_grism(grism, data, detector)
    dispersed_sum = np.sum(simulation)

    assert expected_sum == pytest.approx(dispersed_sum)
