from jinja2 import Environment, FileSystemLoader
from pathlib import Path
 
def generate_report(metrics, model_name, validation_date, dataset_info,             
  output_path, fairness=None, stability=None, drift=None, challenger=None):
    """                                                                                 
    Render a validation report to an HTML file using a Jinja2 template.
                                                                                      
    Parameters
    ----------                                                                          
    metrics : dict  
        Nested dict of computed validation metrics (e.g. AUC, MAE, PSI).
    model_name : str                                                                    
        Name of the model being validated.
    validation_date : str                                                               
        Date of the validation run (e.g. "2026-03-19").
    dataset_info : dict                                                                 
        Metadata about the validation dataset (e.g. n_observations, date_range).
    output_path : str                                                                   
        File path where the rendered HTML report will be written.
    fairness : dict, optional
        Fairness metric results (disparate_impact_ratio, demographic_parity_difference,
        equalized_odds). If None, fairness section is omitted from the report.
    stability : pd.DataFrame, optional
        Output of performance_over_time(). If None, stability section is omitted.
    drift : dict, optional
        PSI value and flag for population drift. If None, drift section is omitted.
    challenger : dict, optional
        Champion vs challenger comparison results. If None, challenger section isvomitted.                  
    """                                                                                
    env = Environment(loader=FileSystemLoader(Path(__file__).parent.parent.parent / "templates"))
    template = env.get_template("validation_report.html")
    html = template.render(
                           metrics = metrics, 
                           model_name = model_name, 
                           validation_date =validation_date, 
                           dataset_info = dataset_info,
                           fairness=fairness, 
                           stability=stability, 
                           drift=drift, 
                           challenger=challenger
            )
    with open(output_path, "w") as f:
        f.write(html)