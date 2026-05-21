"""Tests for GreenCreditTreatment and TreatmentType."""
import pytest
from gcef.treatment import GreenCreditTreatment, TreatmentType, ConditionalityMechanism, VerificationMethod
from gcef.exceptions import BlendedTreatmentNotImplementedError


def make_treatment(**kwargs):
    defaults = dict(
        type=TreatmentType.RATE_REDUCTION,
        conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
        verification_method=VerificationMethod.DOCUMENT_REVIEW,
        intensity=0.03,
    )
    defaults.update(kwargs)
    return GreenCreditTreatment(**defaults)


def test_valid_treatment_instantiates():
    t = make_treatment()
    assert t.type == TreatmentType.RATE_REDUCTION
    assert t.intensity == 0.03


def test_blended_raises_not_implemented():
    with pytest.raises(BlendedTreatmentNotImplementedError):
        make_treatment(type=TreatmentType.BLENDED)


def test_self_reported_verification_flags_warning():
    t = make_treatment(verification_method=VerificationMethod.SELF_REPORTED)
    assert t.issues_self_reported_warning is True


def test_sector_restricted_is_selection_dominant():
    t = make_treatment(type=TreatmentType.SECTOR_RESTRICTED)
    assert t.is_selection_dominant is True


def test_rate_reduction_not_selection_dominant():
    t = make_treatment(type=TreatmentType.RATE_REDUCTION)
    assert t.is_selection_dominant is False


def test_description_includes_type():
    t = make_treatment()
    desc = t.to_description()
    assert "rate_reduction" in desc
