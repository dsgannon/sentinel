from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def model_card(
    model_name,
    model_version,
    model_type,
    owner,
    intended_use,
    training_data,
    performance_metrics,
    fairness=None,
    limitations=None,
    output_path="model_card.html"
):
    """
    Generate a Model Card as an HTML file.

    Model cards are structured documents that communicate a model's
    intended use, performance, fairness characteristics, and limitations
    to technical and non-technical stakeholders. Increasingly expected
    by regulators alongside SR 11-7 validation documentation.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_version : str
        Version identifier (e.g. "2.1.0").
    model_type : str
        Model type (e.g. "classification", "regression").
    owner : str
        Name of the model owner or responsible team.
    intended_use : dict
        Keys: "primary_use" (str), "out_of_scope" (str).
    training_data : dict
        Keys: "source", "n_observations", "date_range", "features".
    performance_metrics : dict
        Computed validation metrics.
    fairness : dict, optional
        Fairness metric results. If None, fairness section is omitted.
    limitations : str, optional
        Known limitations and caveats. If None, placeholder is shown.
    output_path : str, default="model_card.html"
        File path where the rendered HTML report will be written.
    """                                                                                
    env = Environment(loader=FileSystemLoader(Path(__file__).parent.parent.parent / "templates"))
    template = env.get_template("model_card.html")
    html = template.render(
        model_name = model_name,
        model_version = model_version,
        model_type = model_type,
        owner = owner,
        intended_use = intended_use,
        training_data = training_data,
        performance_metrics = performance_metrics,
        fairness = fairness,
        limitations = limitations
    )
    with open(output_path, "w") as f:
        f.write(html)