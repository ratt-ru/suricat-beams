"""Shared fixtures for the synthetic tests."""
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation


@pytest.fixture(autouse=True)
def meerkat_offline(monkeypatch):
    """BeamWizard looks up EarthLocation.of_site("MeerKAT"), which needs astropy's site
    registry: downloaded, or cached from an earlier download. Give it MeerKAT's
    position directly, so the tests run offline and in a fresh environment."""
    meerkat = EarthLocation.from_geodetic(21.443888889 * u.deg, -30.711055553 * u.deg, 1086.6 * u.m)
    monkeypatch.setattr(EarthLocation, "of_site", classmethod(lambda cls, name, *a, **kw: meerkat))
