import json
import datasets
from datasets import DownloadManager, DatasetInfo
import pickle
from PIL import Image, UnidentifiedImageError
import os

class TransferMarket(datasets.GeneratorBasedBuilder):

    def _info(self) -> DatasetInfo:
        """
            info方法，定义数据集的信息，这里要对数据的字段进行定义
        :return:
        """
        return datasets.DatasetInfo(
            description="transfer_market",
            features=datasets.Features({
                    "image_name": datasets.Value("string"),
                    "image_id": datasets.Value("int32"),
                    "cam_id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "image": datasets.Image(),
                    
                }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """
            返回datasets.SplitGenerator
            涉及两个参数：name和gen_kwargs
            name: 指定数据集的划分
            gen_kwargs: 指定要读取的文件的路径，与_generate_examples的入参数一致
        :param dl_manager:
        :return: [ datasets.SplitGenerator ]
        """
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "./market-1501/transfer_market.json"})]

    def _generate_examples(self, filepath):
        """
            生成具体的样本，使用yield
            需要额外指定key，id从0开始自增就可以
        :param filepath:
        :return:
        """
        # Yields (key, example) tuples from the dataset
        idx = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for annot in data["annotations"]:
                stored_text = None
                img_name = annot["image_name"][:-3]
                if os.path.exists('/cluster/home/guanmengyuan/vilt-transreid/vector/market_1039_text_pkl_no_env/'+img_name+'pkl'): #market_1029_text_embeds_pkl_no_env
                    
                    with open('/cluster/home/guanmengyuan/vilt-transreid/vector/market_1039_text_pkl_no_env/'+img_name+'pkl', "rb") as fIn:
                        stored_data = pickle.load(fIn)
                        stored_text = stored_data["text"]
                market_dir = '/cluster/home/guanmengyuan/transfer/market/'
                img = Image.open(market_dir+annot["image_name"]).convert("RGB")
                
                
                yield idx, {
                    "image_name": annot["image_name"],
                    "image_id": annot["image_id"],
                    "cam_id": annot["cam_id"],
                    "text": stored_text,
                    "image": img,
                    
                    
                }
                idx += 1