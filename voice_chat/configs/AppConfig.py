from omegaconf import OmegaConf
import os
from dotenv import load_dotenv

load_dotenv()

# Load the api configurations
Configurations = OmegaConf.load(
    os.path.join(
        os.environ["APPLICATION_ROOT_FOLDER"],
        os.environ["API_CONFIG_PATH"],
        "configs.yaml",
    )
)
