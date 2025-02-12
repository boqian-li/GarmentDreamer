import sys
import yaml

sys.path.append("..")

from trainers.encdec import EncoderDecoderTrainer


def main(config) -> None:
    trainer = EncoderDecoderTrainer(config)
    trainer.train()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cfg_file = sys.argv[1]
        del sys.argv[1]
    else:
        assert 1==0
    with open(run_cfg_file, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
