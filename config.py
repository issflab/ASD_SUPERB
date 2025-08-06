from dataclasses import dataclass
from typing import Optional, Literal
import os

@dataclass
class Config:
    #'aasist', 'sls', or 'xlsrmamba'
    model_arch: Literal['aasist', 'sls', 'xlsrmamba'] = 'xlsrmamba'

    # Dataset name
    # name this variable based on datasets being used to train the models
    # CodecFake = codec
    # Famous Figures = FF
    # ASVspoof 2019 = ASV19
    # ASVspoof 2025 = ASV5
    # FakeXpose = FX
    # In the Wild = ITW
    # DFADD = DFADD
    # MLAAD = MLAAD
    # SpoofCeleb = SpoofCeleb
    # example, data_name = 'codec_FF_ASV19_MLAAD'
    # data_name = 'ASV19_CodecTTS_FF_MLAAD'
    # data_name = 'mlaad_spoofceleb_FF'
    # data_name = 'Codec_FF_ITW_Pod_mlaad_spoofceleb'
    dataset: str = 'ASV19'

    database_path: str = '/data/Data'   # root that contains e.g. spoofceleb/flac/...
    protocols_path: str = '/data/Data'  

    train_protocol: str = 'ASVspoof2019.LA.cm.train.trn.txt'
    dev_protocol: str = 'ASVspoof2019.LA.cm.dev.trl.txt'

    mode: Literal['train', 'eval'] = 'train'

    save_dir: str = '/data/ssl_anti_spoofing/asd_superb/'
    model_name: str = 'run1'

    cuda_device: str = 'cuda:0'

    pretrained_checkpoint: Optional[str] = None

    @property
    def train_protocol_path(self) -> str:
        return os.path.join(self.protocols_path, self.train_protocol)

    @property
    def dev_protocol_path(self) -> str:
        return os.path.join(self.protocols_path, self.dev_protocol)

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.save_dir, self.model_name)

    def prepare_dirs(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)


cfg = Config()

cfg.model_arch = os.getenv('SSL_MODEL_ARCH', cfg.model_arch)
cfg.database_path = os.getenv('SSL_DATABASE_PATH', cfg.database_path)
cfg.protocols_path = os.getenv('SSL_PROTOCOLS_PATH', cfg.protocols_path)
cfg.mode = os.getenv('SSL_MODE', cfg.mode)
cfg.model_name = os.getenv('SSL_MODEL_NAME', cfg.model_name)
cfg.cuda_device = os.getenv('CUDA_DEVICE', cfg.cuda_device)
env_ckpt = os.getenv('SSL_PRETRAINED_CHECKPOINT')
if env_ckpt:
    cfg.pretrained_checkpoint = env_ckpt

cfg.prepare_dirs()
