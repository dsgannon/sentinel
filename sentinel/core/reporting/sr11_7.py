from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def sr11_7_report(
    model_name,
    validation_date,
    dataset_info,
    performance_metrics,
    challenger_results=None,
    narrative=None,
    output_path="sr11_7_report.html"
):
    """                                                                                 
    Generate an SR 11-7 compliant model validation report as an HTML file.

    Auto-populates quantitative sections (performance results, challenger
    benchmarking) from validation run outputs. Narrative sections are
    populated from the narrative dict or left as placeholders for human
    input or LLM fill-in (Phase 6).

    Parameters
    ----------
    model_name : str
        Name of the model being validated.
    validation_date : str
        Date of the validation run (e.g. "2026-03-27").
    dataset_info : dict
        Metadata about the validation dataset (e.g. n_observations, date_range).
    performance_metrics : dict
        Computed validation metrics - AUC, KS, Gini, MAE, PSI, fairness, etc.
    challenger_results : dict, optional
        Output of delong_test() or compare_auc(). If None, section 5 is omitted.
    narrative : dict, optional
        Human-written text for narrative sections. Expected keys:
        "model_purpose", "conceptual_soundness", "data_quality",
        "limitations", "monitoring_plan", "conclusions".
        Missing keys render as placeholder text.
    output_path : str, default="sr11_7_report.html"
        File path where the rendered HTML report will be written.
    """                                                                                
    env = Environment(loader=FileSystemLoader(Path(__file__).parent.parent.parent / "templates"))
    template = env.get_template("sr11_7_report.html")
    html = template.render(
        model_name=model_name,
        validation_date=validation_date,
        dataset_info=dataset_info,
        performance_metrics=performance_metrics,
        challenger_results=challenger_results,
        narrative=narrative
    )
    with open(output_path, "w") as f:
        f.write(html)