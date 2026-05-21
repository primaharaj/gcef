"""gcef.testing — synthetic data and test utilities."""
from gcef.testing.synthetic import (
    make_panel,
    make_valid_panel,
    make_single_lender,
    make_weak_instrument,
    make_marginal_instrument,
    make_short_panel,
    make_low_complier_share,
    make_missing_geocoding,
    make_admin_unit_only,
    make_self_reported_treatment,
    make_adoption_anomaly,
    make_exclusion_violation,
    make_underidentified_outcome,
    make_high_heterogeneity,
    describe_panel,
    SyntheticConfig,
)

__all__ = [
    "make_panel", "make_valid_panel", "make_single_lender",
    "make_weak_instrument", "make_marginal_instrument",
    "make_short_panel", "make_low_complier_share",
    "make_missing_geocoding", "make_admin_unit_only",
    "make_self_reported_treatment", "make_adoption_anomaly",
    "make_exclusion_violation", "make_underidentified_outcome",
    "make_high_heterogeneity", "describe_panel", "SyntheticConfig",
]
