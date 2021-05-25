import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from src.training import train_pipeline
from src.inference import inference_pipeline


@hydra.main(config_path="configs", config_name="config.yaml")
def app(cfg: DictConfig):
    """
    runs pipeline
    """

    cfg_train = cfg.get("train", None)
    cfg_inference = cfg.get("inference", None)
    if cfg_train is not None:
        train_pipeline(cfg_train)
    elif cfg_inference is not None:
        inference_pipeline(cfg_inference)
    else:
        print(
            "please provide either train config using 'train=<train_config_name>' or inference config using 'inference=<inference_config_name>'"
        )


if __name__ == "__main__":
    app()
