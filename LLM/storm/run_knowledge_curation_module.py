from __future__ import annotations


class STORMWikiRunner(Engine):
    def __init__(self,
                 args: STORMWikiRunnerArguments,
                 lm_configs: STORMWikiLMConfigs,
                 rm):
        self.args = args
        self.lm_configs = lm_configs

        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num
        )    

    def run_knowledge_curation_module(self,
                                ground_truth_url: str = "None",
                                callback_handler: BaseCallbackHandler = None) -> StormInformationTable:
        information_table, conversation_log = \
        self.storm_knowledge_curation_module.research(
            topic=self.topic,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            max_perspective=self.args.max_perspective,
            disable_perspective=False,
            return_conversation_log=True
        )

        # FileIOHelper.dump_json(conversation_log, os.path.join(self.article_output_dir, 'conversation_log.json'))

        # information_table.dump_url_to_info(os.path.join(self.article_output_dir, 'raw_search_results.json'))

        return information_table




class StormKnowledgeCurationModule(KnowledgeCurationModule):
    def __init__():
        # ...
        self.persona_generator = persona_generator
        # ...
        self.conv_simulator = ConvSimulator(...)

        # ...

    def research(self,
                 topic: str,
                 ground_truth_url: str,
                 callback_handler: BaseCallbackHandler,
                 max_perspective: int = 0,
                 disable_perspective: bool = True,
                 return_conversation_log=False) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """
        Returns:
            collected_information
        """

        '''觀點perspective 使用者畫像personas'''
        # identify personas
        callback_handler.on_identify_perspective_start()
        considered_personas = []
        if disable_perspective:
            considered_personas = [""]
        else:
            '''設定對象'''
            considered_personas = self._get_considered_personas(topic=topic, max_num_persona=max_perspective)
                # return
                self.persona_generator.generate_persona(topic=topic, max_num_persona=max_num_persona)
                class StormPersonaGenerator():
                    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
                        default_persona = 'Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic.'
                        personas = self.create_writer_with_persona(topic=topic)
                            #  = CreateWriterWithPersona(engine=engine)
                        considered_personas = [default_persona] + personas.personas[:max_num_persona]
                        return considered_personas



        callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # run conversation 
        callback_handler.on_information_gathering_start()
        '''_run_conversation很複雜'''
        conversations = self._run_conversation(conv_simulator=self.conv_simulator,
                                               topic=topic,
                                               ground_truth_url=ground_truth_url,
                                               considered_personas=considered_personas,
                                               callback_handler=callback_handler)

        information_table = StormInformationTable(conversations)
        callback_handler.on_information_gathering_end()
        if return_conversation_log:
            return information_table, StormInformationTable.construct_log_dict(conversations)
        return information_table


    def _get_considered_personas():
    def _run_conversation(self, 
                        #   conv_simulator , 
                          conv_simulator : ConvSimulator, 
                          topic, 
                          ground_truth_url, 
                          considered_personas,
                          callback_handler: BaseCallbackHandler
                        ) -> List[Tuple[str, List[DialogueTurn]]]:
        
        conversations = []

        def run_conv(persona):
            # return conv_simulator(
            return self.conv_simulator(
                topic=topic,
                ground_truth_url=ground_truth_url,
                persona=persona,
                callback_handler=callback_handler
            )

        max_workers = min(self.max_thread_num, len(considered_personas))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_persona = { executor.submit(run_conv, persona): persona 
                                    for persona in considered_personas
                                }
            if streamlit_connection:
                # ...

            for future in as_completed(future_to_persona):
                persona = future_to_persona[future]
                conv = future.result()
                conversations.append((persona, ArticleTextProcessing.clean_up_citation(conv).dlg_history))

        return conversations

class StormInformationTable(InformationTable):
    def __init__(self, conversations=List[Tuple[str, List[DialogueTurn]]]):
        super().__init__()
        self.conversations = conversations
        self.url_to_info: Dict[str, StormInformation] = StormInformationTable.construct_url_to_info(self.conversations)
        
    def construct_url_to_info(conversations: List[Tuple[str, List[DialogueTurn]]]) -> Dict[str, StormInformation]:
        url_to_info = {}
        for (persona, conv) in conversations:
            for turn in conv:
                for storm_info in turn.search_results:
                    #  url_to_info[storm_info.url] = ...
        # ...
        return url_to_info
