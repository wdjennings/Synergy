import mlflow

from src.global_config import GlobalConfig
from src.utils.mlflow.load_runs import load_runs
from src.utils.mlflow.vs_parameters import HeatmapTypes, plot_vs_parameters

mlflow.set_experiment(GlobalConfig().experiment_name)

runs_df = load_runs(cached=False)

plot_vs_parameters(
    runs_df=runs_df,
    parameter_grid=GlobalConfig().parameter_grid,
    type=HeatmapTypes.Epidemics1D,
)


# for type in HeatmapTypes.all():
#     plot_parameters_heatmap(
#         runs_df=runs_df,
#         parameter_grid=GlobalConfig().parameter_grid,
#         type=type,
#     )
