import os
import datasets


class CustomNERDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B_T", "I_T", "B_LOC", "I_LOC", "B_ORG", "I_ORG", "B_PER", "I_PER"]
                        )
                    ),
                    "id": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = "/ner-data"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": (os.path.join(data_dir, "train.txt"), os.path.join(data_dir, "train_TAG.txt"))}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": (os.path.join(data_dir, "dev.txt"), os.path.join(data_dir, "dev_TAG.txt"))}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": (os.path.join(data_dir, "test.txt"), os.path.join(data_dir, "test_TAG.txt"))}
            ),
        ]

    def _generate_examples(self, filepaths):
        sentences_file, tags_file = filepaths
        with open(sentences_file, "r", encoding="utf-8") as sf, open(tags_file, "r", encoding="utf-8") as tf:
            for idx, (sentence, tags) in enumerate(zip(sf, tf)):
                tokens = sentence.strip().split()
                ner_tags = tags.strip().split()
                assert len(tokens) == len(ner_tags), f"Tokens and tags length mismatch at line {idx}."
                yield idx, {
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "id": str(idx)
                }
