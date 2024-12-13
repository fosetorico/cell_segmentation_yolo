import os,sys
import shutil
import zipfile
import yaml
from cellSegmentation.utils.main_utils import read_yaml_file
from cellSegmentation.logger import logging
from cellSegmentation.exception import AppException
from cellSegmentation.entity.config_entity import ModelTrainerConfig
from cellSegmentation.entity.artifacts_entity import ModelTrainerArtifact

current_dir = os.getcwd()

def verify_dataset_structure():
    for path in ["train/images", "valid/images", "test/images"]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset path '{path}' does not exist.")
    print("Dataset structure is valid.")

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def update_data_yaml(self, file_path: str):
        """
        Update paths in data.yaml to ensure they are relative to the current directory.
        """
        try:
            with open(file_path, 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)

            # Update paths to use forward slashes and match directory structure
            data['train'] = "train/images"
            data['val'] = "valid/images"
            data['test'] = "test/images"

            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

            print(f"Validated and updated data.yaml: {data}")
            logging.info(f"Updated data.yaml paths: {data}")
        except Exception as e:
            print(f"Failed to validate or update data.yaml: {e}")
            raise AppException(f"Failed to update data.yaml: {e}", sys)

    def validate_dataset_structure(self):
        """Ensure required directories exist."""
        required_dirs = ["train/images", "valid/images", "test/images"]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise AppException(f"Required directory '{dir_path}' is missing.", sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Unzip the data
            logging.info("Unzipping data")
            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                temp_dir = "temp_unzipped"
                zip_ref.extractall(temp_dir)

                # Move contents of the inner folder to the current working directory
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        for sub_item in os.listdir(item_path):
                            shutil.move(os.path.join(item_path, sub_item), os.getcwd())
                    else:
                        shutil.move(item_path, os.getcwd())

                shutil.rmtree(temp_dir)  # Clean up temporary folder

            os.remove("data.zip")

            # Validate dataset structure
            logging.info("Validating dataset structure")
            self.validate_dataset_structure()

            # Update data.yaml paths
            logging.info("Updating data.yaml paths")
            self.update_data_yaml("data.yaml")

            logging.info("Validating dataset structure")
            verify_dataset_structure()

            # Train the model
            logging.info("Starting YOLO model training")
            training_result = os.system(
                f"yolo task=segment mode=train model={self.model_trainer_config.weight_name} "
                f"data={os.path.join(current_dir, 'data.yaml')} epochs={self.model_trainer_config.no_epochs} imgsz=640 save=true"
            )
            if training_result != 0:
                raise AppException("YOLO training failed.", sys)

            # Save the trained model
            trained_model_path = "runs/segment/train/weights/best.pt"
            if not os.path.exists(trained_model_path):
                raise AppException(f"Trained model not found at {trained_model_path}", sys)

            logging.info("Saving the trained model")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(trained_model_path, self.model_trainer_config.model_trainer_dir)

            # Clean up unnecessary files and directories
            logging.info("Cleaning up temporary files")
            shutil.rmtree("train", ignore_errors=True)
            shutil.rmtree("valid", ignore_errors=True)
            shutil.rmtree("test", ignore_errors=True)
            shutil.rmtree("runs", ignore_errors=True)
            os.remove("yolov8s-seg.pt")
            os.remove("data.yaml")

            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
