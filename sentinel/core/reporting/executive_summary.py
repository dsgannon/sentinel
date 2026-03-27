from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def executive_summary(
    model_name,
    validation_date,
    overall_result,
    key_findings,
    performance_summary,
    recommendation,
    output_path = "executive_summary.html"
):
    """
    Generate a plain English executive summary of a model validation run.

    Translates quantitative validation results into business language
    suitable for non-technical stakeholders (CFO, board, risk committee).
    Complements the SR 11-7 report and model card for regulatory packages.

    Parameters
    ----------
    model_name : str
        Name of the model being summarized.
    validation_date : str
        Date of the validation run.
    overall_result : str
        Validation outcome: "approved", "approved with conditions", or "rejected".
    key_findings : list of str
        Plain English bullet points summarizing the most important findings.
    performance_summary : dict
        Small set of key metrics for executive audience (e.g. AUC, PSI).
    recommendation : str
        Plain English next steps and recommendations.
    output_path : str, default="executive_summary.html"
        File path where the rendered HTML report will be written.
    """

    env = Environment(loader=FileSystemLoader(Path(__file__).parent.parent.parent / "templates"))
    template = env.get_template("executive_summary.html")
    html = template.render(
        model_name = model_name,
        validation_date = validation_date,
        overall_result = overall_result,
        key_findings = key_findings,
        performance_summary = performance_summary,
        recommendation = recommendation
    )
    with open(output_path, "w") as f:
        f.write(html)

