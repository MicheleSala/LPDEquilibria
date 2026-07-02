"""Smoke tests verifying each module imports and exposes its expected class."""

import pytest


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("coils", "Coils"),
        ("equilibrium", "Equilibrium"),
        ("mesh", "Mesh"),
        ("plotting", "PlotEquilibrium"),
        ("resonances", "Resonances"),
        ("sources", "Sources"),
    ],
)
def test_module_exposes_class(module_name, class_name):
    module = __import__(module_name)
    assert hasattr(module, class_name), f"{module_name} missing {class_name}"


def test_import_charged_motion():
    import ChargedMotion  # noqa: F401


def test_charged_motion_exposes_expected_classes():
    import ChargedMotion

    for name in ("Particle", "ParticlePusher", "MagneticField"):
        assert hasattr(ChargedMotion, name), f"ChargedMotion module missing {name}"
