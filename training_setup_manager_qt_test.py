def parse_param_list(param_value, convert_func=float):
    """Safely parse parameter that might be comma-separated."""
    if isinstance(param_value, (int, float)):
        return [param_value]
    try:
        values = [v.strip() for v in str(param_value).replace(',', ' ').split() if v]
        if not values:
            raise ValueError(f"Empty parameter value")
        return [convert_func(v) for v in values]
    except ValueError as e:
        self.logger.error(f"Error parsing parameter: {param_value}")
        raise ValueError(f"Invalid value in list: {param_value}. Expected {convert_func.__name__} values.")

# Parse parameters safely - these will be lists even if single values
learning_rates = parse_param_list(self.current_hyper_params['INITIAL_LR'], float)
train_val_splits = [float(self.current_hyper_params['TRAIN_VAL_SPLIT'])]  # This should not be a list typically
lr_params = parse_param_list(self.current_hyper_params['LR_PARAM'], float)
lr_periods = parse_param_list(self.current_hyper_params['LR_PERIOD'], int)
plateau_patience = parse_param_list(self.current_hyper_params['PLATEAU_PATIENCE'], int)
plateau_factors = parse_param_list(self.current_hyper_params['PLATEAU_FACTOR'], float)
valid_patience_values = parse_param_list(self.current_hyper_params['VALID_PATIENCE'], int) 

# Create tasks for each combination of hyperparameters
for lr in learning_rates:
    for train_val_split in train_val_splits:
        for lr_param in lr_params:
            for drop_period in lr_periods:
                for drop_factor in plateau_factors:
                    for patience in plateau_patience:  # Using plateau_patience instead!
                        for lookback in lookbacks:
                            for batch_size in batch_sizes: 