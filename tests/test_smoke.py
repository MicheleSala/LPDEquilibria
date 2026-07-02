"""Smoke tests verifying core modules import and basic types are wired up.

Note: ChargedMotion is not covered here because its module body executes a demo
that reads from stdin. Wrap the demo in `if __name__ == "__main__":` to make
the module importable, then extend these tests.
"""


def test_import_equilibrium():
    import Equilibrium  # noqa: F401


def test_equilibrium_exposes_expected_classes():
    import Equilibrium

    for name in ("Coils", "Equilibrium", "Mesh", "PlotEquilibrium", "Resonances", "Sources"):
        assert hasattr(Equilibrium, name), f"Equilibrium module missing {name}"
