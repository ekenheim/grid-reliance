"""
Shared pytest fixtures for grid-resilience tests.

Uses pymc.testing.mock_sample to replace pm.sample with prior predictive
sampling so that model structure tests run in seconds, not minutes.
"""

import pytest
from pymc.testing import mock_sample_setup_and_teardown

mock_pymc_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)
