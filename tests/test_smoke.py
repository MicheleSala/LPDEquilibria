"""Smoke tests verifying core modules import and basic types are wired up."""


def test_import_equilibrium():
    import Equilibrium  # noqa: F401


def test_import_charged_motion():
    import ChargedMotion  # noqa: F401


def test_equilibrium_exposes_expected_classes():
    import Equilibrium

    for name in ("Coils", "Equilibrium", "Mesh", "PlotEquilibrium", "Resonances", "Sources"):
        assert hasattr(Equilibrium, name), f"Equilibrium module missing {name}"


def test_charged_motion_exposes_expected_classes():
    import ChargedMotion

    for name in ("Particle", "ParticlePusher", "MagneticField"):
        assert hasattr(ChargedMotion, name), f"ChargedMotion module missing {name}"
