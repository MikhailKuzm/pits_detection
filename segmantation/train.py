import hydra
from hydra.utils import instantiate
import logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='train')
def train(cfg): 
    model = instantiate(cfg['model']) 
    trainer = instantiate(cfg['trainer']) 
    trainer.fit(model)
    
if __name__ == '__main__':
    train()
    