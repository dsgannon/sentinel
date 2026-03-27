def assign_tier(model_type, purpose, annual_impact_usd=None, regulatory_use=False):
    """
    Auto-classify a model into SR 11-7 risk tiers (1, 2, or 3).

    Tier 1 — High risk: drives significant financial decisions, large dollar
    impact, or used for regulatory reporting. Requires full validation and
    annual review.

    Tier 2 — Moderate risk: meaningful but not critical impact. Periodic
    validation required.

    Tier 3 — Low risk: minimal financial impact, internal analytics only.
    Light-touch validation sufficient.

    Parameters
    ----------
    model_type : str
        Type of model (e.g. "classification", "regression", "pricing",
        "capital", "analytics", "reporting").
    purpose : str
        Description of the model's business purpose (informational).
    annual_impact_usd : float, optional
        Estimated annual dollar impact of decisions driven by the model.
        If None, dollar impact is not used in tiering.
    regulatory_use : bool, default=False
        True if the model is used for regulatory reporting or capital
        calculations.

    Returns
    -------
    int
        1, 2, or 3 representing the assigned risk tier.
    """
    if regulatory_use or ( annual_impact_usd is not None and annual_impact_usd > 10_000_000) or model_type in ("pricing", "capital"):
        return 1
    elif not regulatory_use False and ( annual_impact_usd is None or annual_impact_usd < 1_000_000) and model_type in ("analytics", "reporting"):
        return 3
    else:
        return 2