# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/26 7:55 PM
# @File: llm_classification_model
# @Email: mlshenkai@163.com
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger
import numpy as np


class LLMClassificationModel:
    def __init__(
        self, model_name, use_cuda=True, cuda_device=-1, quantization_level=None, cache_dir=None,
    ):
        """
        init LLM model
        Args:
            model_name: model name need in (ChatGML)
            use_cuda:
            cuda_device:
            quantization_level:
        """

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).half()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._move_model_to_device()
        self.model_name = model_name
        self.is_init_prompts = False

    def _move_model_to_device(self):
        self.model.to(self.device)

    def init_prompts(self, class_examples: dict, class_list=None):
        """
        初始化前置prompt，便于模型做 incontext learning。
        Args:
            class_examples:
            class_list:

        Returns:

        """
        # # class_list = list(class_examples.keys())
        # class_list =
        if class_list is None:
            class_list_set = set()
            for key in list(class_examples.keys()):
                class_list_set.add(key)
            class_list = list(class_list_set)
        pre_history = [(f"现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：{class_list}类别中。", f"好的。")]

        for _type, example in class_examples.items():
            pre_history.append((f"“{example}”是 {class_list} 里的什么类别？", _type))
        self.custom_settings = {"class_list": class_list, "pre_history": pre_history}
        self.is_init_prompts = True

    def predict(self, to_predict: list):
        if not self.is_init_prompts:
            logger.error(f"llm {self.model_name} is not trained...")
            return

        preds = []
        for sentence_idx, sentence in enumerate(to_predict):
            sentence_with_prompt = (
                f"“{sentence}”是 {self.custom_settings['class_list']} 里的什么类别？"
            )
            response, history = self.model.chat(
                self.tokenizer,
                sentence_with_prompt,
                history=self.custom_settings["pre_history"],
            )
            preds.append(response)
        return preds


if __name__ == "__main__":
    llm_model = LLMClassificationModel(model_name="/code-online/glm_5b_model")
    class_examples = {
        '人物': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员 [1]  。2005年，首次登台演出。2012年，主演卢卫国执导的喜剧电影《就是闹着玩的》。2013年在北京举办相声专场。',
        '书籍': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成，第一部于2006年5月起在《科幻世界》杂志上连载，第二部于2008年5月首次出版，第三部则于2010年11月出版。',
        '电视剧': '《狂飙》是由中央电视台、爱奇艺出品，留白影视、中国长安出版传媒联合出品，中央政法委宣传教育局、中央政法委政法综治信息中心指导拍摄，徐纪周执导，张译、张颂文、李一桐、张志坚、吴刚领衔主演，倪大红、韩童生、李建义、石兆琪特邀主演，李健、高叶、王骁等主演的反黑刑侦剧。',
        '电影': '《流浪地球》是由郭帆执导，吴京特别出演、屈楚萧、赵今麦、李光洁、吴孟达等领衔主演的科幻冒险电影。影片根据刘慈欣的同名小说改编，故事背景设定在2075年，讲述了太阳即将毁灭，毁灭之后的太阳系已经不适合人类生存，而面对绝境，人类将开启“流浪地球”计划，试图带着地球一起逃离太阳系，寻找人类新家园的故事。',
        '城市': '乐山，古称嘉州，四川省辖地级市，位于四川中部，四川盆地西南部，地势西南高，东北低，属中亚热带气候带；辖4区、6县，代管1个县级市，全市总面积12720.03平方公里；截至2021年底，全市常住人口315.1万人。',
        '国家': '瑞士联邦（Swiss Confederation），简称“瑞士”，首都伯尔尼，位于欧洲中部，北与德国接壤，东临奥地利和列支敦士登，南临意大利，西临法国。地处北温带，四季分明，全国地势高峻，矿产资源匮乏，森林及水力资源丰富，总面积41284平方千米，全国由26个州组成（其中6个州为半州）。'
    }
    llm_model.init_prompts(class_examples)
    print(llm_model.predict(["张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地男演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。"]))