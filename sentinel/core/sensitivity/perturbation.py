import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def tornado_diagram(sensitivity_df, metric_name = "Metric", figsize = (10, 6)):
    """
    Plot a tornado diagram showing feature sensitivity to perturbation.             
                                                                                    
    Visualizes the output of feature_sensitivity() as horizontal bars,              
    sorted by absolute metric change. Features causing the largest metric           
    shift appear at the top. Positive deltas indicate the metric worsened           
    when the feature was perturbed; negative deltas indicate improvement.           
                                                                                    
    Parameters                                                                      
    ----------  
    sensitivity_df : pd.DataFrame                                                   
        Output of feature_sensitivity(). Must contain "feature" and
        "metric_delta" columns.                                                     
    metric_name : str, default="Metric"
        Label for the x-axis and plot title.                                        
    figsize : tuple, default=(10, 6)
        Matplotlib figure size (width, height) in inches.                           
                                                                                    
    Returns
    -------                                                                         
    matplotlib.figure.Figure
        The tornado diagram figure. Call fig.savefig() to export.
    """  
    df = sensitivity_df.sort_values("metric_delta", key=abs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df["feature"], df["metric_delta"])
    ax.axvline(x=0, color = "black", linewidth = 0.8)
    ax.set_xlabel(metric_name) 
    ax.set_title(f"Tornado Diagram — {metric_name} Sensitivity")    
    plt.tight_layout()
    return fig