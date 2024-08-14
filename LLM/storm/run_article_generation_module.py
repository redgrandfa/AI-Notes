from __future__ import annotations


class STORMWikiRunner(Engine):
    def __init__(self,
                args: STORMWikiRunnerArguments,
                lm_configs: STORMWikiLMConfigs,
                rm):
        self.args = args
        self.lm_configs = lm_configs

        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num
        )

    def run_article_generation_module(self,
                                      outline: StormArticle,
                                      information_table=StormInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> StormArticle:

        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler
        )
        # draft_article.dump_article_as_plain_text(... 'storm_gen_article.txt')
        # draft_article.dump_reference_to_file(... 'url_to_info.json')
        return draft_article


class StormArticleGenerationModule(ArticleGenerationModule):
    def __init__(self,
                 article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 retrieve_top_k: int = 5,
                 max_thread_num: int = 10):
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)


    def generate_article(self,
                         topic: str,
                         information_table: StormInformationTable,
                         article_with_outline: StormArticle,
                         callback_handler: BaseCallbackHandler = None) -> StormArticle:
        """
        based on the 
            information table 
        and 
            article outline.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        sections_to_write = article_with_outline.get_first_level_section_names()
                                            # 第一層標題 return [i.section_name for i in self.root.children]



        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(f'No outline for {topic}. Will directly search with the topic.')
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic]
            )
            section_output_dict_collection = [section_output_dict]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    '''複雜 資料處理'''

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())




        article: StormArticle = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(parent_section_name=topic,
                                   current_section_content=section_output_dict["section_content"],
                                   current_section_info_list=section_output_dict["collected_info"])
        article.post_processing()
        return article


    def generate_section(self, topic, section_name, information_table, section_outline, section_query):
        collected_info: List[StormInformation] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(queries=section_query,
                                                                    search_top_k=self.retrieve_top_k)
        
        # 傳參數給物件，猜測是執行 ConvToSection.forward
        output = self.section_gen(topic=topic,
                                  outline=section_outline,
                                  section=section_name,
                                  collected_info=collected_info)
        

        return {"section_name": section_name, 
                "section_content": output.section, 
                "collected_info": collected_info}


class ConvToSection(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(self, topic: str, outline: str, section: str, collected_info: List[StormInformation]):
        # ...
        return dspy.Prediction(section=section)


class StormInformationTable(InformationTable):
    def prepare_table_for_retrieval(self):
        '''寫上一堆物件屬性'''
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.collected_urls = []
        self.collected_snippets = []
        for url, information in self.url_to_info.items():
            for snippet in information.snippets:
                self.collected_urls.append(url)
                self.collected_snippets.append(snippet)
        self.encoded_snippets = self.encoder.encode(self.collected_snippets, show_progress_bar=False)


class StormArticle(Article):
    def get_first_level_section_names(self) -> List[str]:
        return [i.section_name for i in self.root.children]
    def update_section(self,
                       current_section_content: str,
                       current_section_info_list: List[StormInformation],
                       parent_section_name: Optional[str] = None) -> Optional[ArticleSectionNode]:
        """
        Add new section to the article. 
        複雜
        """
        if current_section_info_list is not None:
            pass

        article_dict = ArticleTextProcessing.parse_article_into_dict(current_section_content)
        self.insert_or_create_section(article_dict, ...)



    def post_processing(self):
        self.prune_empty_nodes()
        self.reorder_reference_index()

