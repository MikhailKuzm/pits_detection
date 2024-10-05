import hydra
from hydra.utils import instantiate
import logging
#log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='pytorch_train')
def train(cfg):  
    model = instantiate(cfg['model']) 
    trainer = instantiate(cfg['trainer'], callbacks = [instantiate(cfg.callbacks[clb_name]) for clb_name in cfg.callbacks]) 
    trainer.fit(model)
    
if __name__ == '__main__':
    train()
    