from __future__ import annotations


class STORMWikiRunner(Engine):
    def __init__(self,
                args: STORMWikiRunnerArguments,
                lm_configs: STORMWikiLMConfigs,
                rm):
        self.args = args
        self.lm_configs = lm_configs

        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        

    def run_outline_generation_module(self,
                                      information_table: StormInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> StormArticle:

        outline, draft_outline = \
        self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler
        )
        # outline.dump_outline_to_file( ...'storm_gen_outline.txt')
        # draft_outline.dump_outline_to_file( ... "direct_gen_outline.txt")
        return outline

class StormOutlineGenerationModule(OutlineGenerationModule):
    def __init__(self,
                 outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)

    def generate_outline(self,
                         topic: str,
                         information_table: StormInformationTable,
                         old_outline: Optional[StormArticle] = None,
                         callback_handler: BaseCallbackHandler = None,
                         return_draft_outline=False
                        ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:

        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        concatenated_dialogue_turns = sum([conv for (_, conv) in information_table.conversations], [])
        '''pass參數給WriteOutline物件 是??
            參數看起來像 執行WriteOutline.forward方法
        '''
        result = self.write_outline(topic=topic, dlg_history=concatenated_dialogue_turns,
                                    callback_handler=callback_handler)
        article_with_outline_only = StormArticle.from_outline_str(topic=topic, outline_str=result.outline)
        article_with_draft_outline_only = StormArticle.from_outline_str(topic=topic,
                                                                        outline_str=result.old_outline)
        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page."""
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        # super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine


    def forward(self, topic: str, dlg_history, old_outline: Optional[str] = None,
                callback_handler: BaseCallbackHandler = None):
        '''...複雜'''
        return dspy.Prediction(outline=outline, old_outline=old_outline)


class StormArticle(Article):
    def from_outline_str(cls, topic: str, outline_str: str):
        '''複雜'''
        lines = []
        # ...
        instance = cls(topic)
        # ...
        return instance
