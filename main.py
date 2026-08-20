from sign_language_detection.configuration import ConfigurationManager
from sign_language_detection.components.data_ingestion import DataIngestion
from sign_language_detection.components.data_validation import DataValidation


def main():

    config_manager = ConfigurationManager()

    data_ingestion_config = config_manager.get_data_ingestion_config()

    data_ingestion = DataIngestion(
        config=data_ingestion_config
    )

    train_output, val_output, test_output = (
        data_ingestion.initiate_data_ingestion()
    )

    print("\nData Ingestion Completed Successfully")
    print(f"Train metadata      : {train_output}")
    print(f"Validation metadata : {val_output}")
    print(f"Test metadata       : {test_output}")

    data_validation_config = config_manager.get_data_validation_config()
    
    data_validation = DataValidation(
        config=data_validation_config
    )

    data_validation_output = (
        data_validation.initiate_data_validation()
    )

    print("\nData Validation Completed Successfully")

if __name__ == "__main__":
    main()