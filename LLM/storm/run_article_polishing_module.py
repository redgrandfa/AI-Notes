from __future__ import annotations
import copy


class STORMWikiRunner(Engine):
    def __init__(self,
                args: STORMWikiRunnerArguments,
                lm_configs: STORMWikiLMConfigs,
                rm):
        self.args = args
        self.lm_configs = lm_configs

        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm
        )



    def run_article_polishing_module(self,
                                     draft_article: StormArticle,
                                     remove_duplicate: bool = False) -> StormArticle:

        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate
        )
        # FileIOHelper.write_str(polished_article.to_string(),... 'storm_gen_article_polished.txt')
        return polished_article



class StormArticlePolishingModule(ArticlePolishingModule):
    def __init__(self,
                 article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm,
            polish_engine=self.article_polish_lm
        )

    def polish_article(self,
                       topic: str,
                       draft_article: StormArticle,
                       remove_duplicate: bool = False) -> StormArticle:

        # 猜是 執行 PolishPageModule.forward
        polish_result = self.polish_page(topic=topic, 
                                         draft_page = draft_article.to_string(), polish_whole_page=remove_duplicate)

        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            '\n\n'.join(
                [f"# summary\n{polish_result.lead_section}", 
                polish_result.page]
            )
        )

        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()
        return polished_article


class PolishPageModule(dspy.Module):
    def __init__(self, 
                 write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        pass

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # ...
        return dspy.Prediction(lead_section=lead_section, page=page)

class StormArticle(Article):
    def insert_or_create_section(self, 
                                 article_dict: Dict[str, Dict], 
                                 parent_section_name: str = None,
                                 trim_children=False):
        parent_node = self.root if parent_section_name is None \
                                else self.find_section(self.root, parent_section_name)

        if trim_children:
            for child in parent_node.children[:]:
                if ...:
                    parent_node.remove_child(child)

        for section_name, content_dict in article_dict.items():
            '''
            複雜
                parent_node.add_child...
            '''
            # recursively 
            self.insert_or_create_section(article_dict=content_dict["subsections"], 
                                          parent_section_name=section_name,
                                          trim_children=True)

    def post_processing(self):
        self.prune_empty_nodes()
        self.reorder_reference_index()
