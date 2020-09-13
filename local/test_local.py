from vocab import Vocabulary
import evaluation

evaluation.evalrank("models/model_best.pth.tar", data_path="../data/", split="test")