"""Tracer bullet: package is installable and exposes version."""

import stupid_cat


def test_package_has_version():
    assert stupid_cat.__version__
