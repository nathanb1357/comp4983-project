simulation_configs = {
    "default": {
        "dataset": {
            "path": "./original_data/trainingset.csv",
            "target_column": "ClaimAmount",
            "drop_columns": ["rowIndex"]
        },
        "models": {
            "RandomForest": {
                "pipeline": {
                    "scaler": True,  # Whether to use a scaler
                    "model": "RandomForestRegressor"
                },
                "params": {
                    "model__n_estimators": [10, 20],
                    "model__max_depth": [2, 5],
                    "model__min_samples_split": [2, 5]
                }
            }
        },
        "metrics": ["MSE", "MAE", "RMSE", "R2"],
        "output_pdf": "performance_tuning_default.pdf"
    },
    "svr_simulation": {
        "dataset": {
            "path": "./original_data/trainingset.csv",
            "target_column": "ClaimAmount",
            "drop_columns": ["rowIndex"]
        },
        "models": {
            "SVR": {
                "pipeline": {
                    "scaler": True,
                    "model": "SVR"
                },
                "params": {
                    "model__C": [0.1, 1, 10],
                    "model__kernel": ["linear", "rbf"]
                }
            }
        },
        "metrics": ["MSE", "MAE", "R2"],
        "output_pdf": "performance_tuning_svr.pdf"
    },
    "single-model-exploration-1": {
        "dataset": {
            "path": "./original_data/trainingset.csv",
            "target_column": "ClaimAmount",
            "drop_columns": ["rowIndex"]
        },
        "models": {
            "RandomForest": {
                "pipeline": {
                    "scaler": True,  # Whether to use a scaler
                    "model": "RandomForestRegressor"
                },
                "params": {
                    "model__n_estimators": [10, 20, 30, 40],
                    "model__max_depth": [2, 5, 10],
                    "model__min_samples_split": [2, 5, 10]
                }
            }
        },
        "metrics": ["MSE", "MAE", "RMSE", "R2"],
        "output_pdf": "performance_tuning_default.pdf"
    },
    "milestone1": {
        "dataset": {
            "path": "./original_data/trainingset.csv",
            "target_column": "ClaimAmount",
            "drop_columns": ["rowIndex"]
        },
        "models": {
            "RandomForest": {
                "pipeline": {
                    "scaler": True,
                    "model": "RandomForestRegressor"
                },
                "params": {
                    "model__n_estimators": [50, 100],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5]
                }
            },
            "SVR": {
                "pipeline": {
                    "scaler": True,
                    "model": "SVR"
                },
                "params": {
                    "model__C": [0.1, 1, 10],
                    "model__kernel": ["linear", "rbf"]
                }
            },
            "GradientBoosting": {
                "pipeline": {
                    "scaler": False,
                    "model": "GradientBoostingRegressor"
                },
                "params": {
                    "model__n_estimators": [50, 100],
                    "model__learning_rate": [0.01, 0.1],
                    "model__max_depth": [3, 5]
                }
            },
            "LinearRegression": {
                "pipeline": {
                    "scaler": True,
                    "model": "LinearRegression"
                },
                "params": {
                    "model__fit_intercept": [True, False],
                    "model__normalize": [False]  # Deprecated in newer sklearn, but added for completeness
                }
            },
            "Ridge": {
                "pipeline": {
                    "scaler": True,
                    "model": "Ridge"
                },
                "params": {
                    "model__alpha": [0.1, 1.0, 10.0],
                    "model__solver": ["auto", "saga"]
                }
            },
            "Lasso": {
                "pipeline": {
                    "scaler": True,
                    "model": "Lasso"
                },
                "params": {
                    "model__alpha": [0.1, 1.0, 10.0],
                    "model__max_iter": [1000, 5000]
                }
            }
        },
        "metrics": ["MSE", "MAE", "RMSE", "R2"],
        "output_pdf": "performance_tuning_milestone1.pdf"
    }
}
