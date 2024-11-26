# import os
# # Add the parent directory to the system path
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import pandas as pd
# from .preprocessing import log_transform
# from .train import train_model, save_model
# from .evaluation import evaluate_model
# import logging
# import config

# # Configure logging with datetime
# logging.basicConfig(filename=os.path.join('logs', 'main.log'),
#                     level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# if __name__ == "__main__":
#     # Directory containing CSV files
#     input_dir = config.input_dir

#     # Process each CSV file in the directory
#     for csv_file in os.listdir(input_dir):
#         if csv_file.endswith(".csv"):
#             full_path = os.path.join(input_dir, csv_file)

#             # Load the CSV file
#             df = pd.read_csv(full_path)

#             # Apply log transformation
#             df_log = log_transform(df, ['o_rice', 'h_rice', 'l_rice', 'c_rice'])

#             # Train the model
#             model, history, X_test, y_test = train_model(df_log)

#             # Save the model
#             model_name = f"lstm_model_{os.path.basename(csv_file).split('.')[0]}.keras"
#             save_model(model, model_name)

#             # Evaluate the model
#             evaluate_model(model, history, X_test, y_test, csv_file)

# if __name__ == "__main__":
#     main()
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import pandas as pd
from .preprocessing import log_transform
from .train import train_model, save_model
from .evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging with datetime
logging.basicConfig(filename=os.path.join('logs', 'main.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Directory containing CSV files
    input_dir = config.input_dir

    # Process each CSV file in the directory
    for csv_file in os.listdir(input_dir):
        if csv_file.endswith(".csv"):
            full_path = os.path.join(input_dir, csv_file)

            # Load the CSV file
            df = pd.read_csv(full_path)

            # Apply log transformation
            df_log = log_transform(df, ['o_rice', 'h_rice', 'l_rice', 'c_rice'])

            # Train the model
            model, history, X_test, y_test = train_model(df_log)

            # Save the model
            model_name = f"lstm_model_{os.path.basename(csv_file).split('.')[0]}.keras"
            save_model(model, model_name)

            # # Save feature scaler
            # feature_scaler = StandardScaler()
            # feature_scaler.fit(df[['o_rice', 'h_rice', 'l_rice', 'c_rice']])
            # geo_id = os.path.basename(csv_file).split('.')[0]
            # save_feature_scaler(feature_scaler, geo_id)

            # Evaluate the model
            evaluate_model(model, history, X_test, y_test, csv_file)

if __name__ == "__main__":
    main()
