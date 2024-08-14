
def main():
    lm_configs = STORMWikiLMConfigs()
        # conv_simulator_lm
        # question_asker_lm
        # outline_gen_lm
        # article_gen_lm
        # article_polish_lm

    engine_args = STORMWikiRunnerArguments( 
        output_dir
        max_conv_turn
        max_perspective     #觀點量?
        search_top_k
        max_thread_num      #關係到rate limit
    )
    rm : dspy.Retrieve =  BingSearch/YouRM/BraveRM 之類的

    runner = STORMWikiRunner(engine_args, lm_configs, rm)
    # STORMWikiRunner(Engine)

    runner.run( ''' ''' )
        if do_research:
            information_table: StormInformationTable 
            = run_knowledge_curation_module
                StormKnowledgeCurationModule(KnowledgeCurationModule)
        
        
        if do_generate_outline:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(...)
            outline: StormArticle 
                = self.run_outline_generation_module(information_table,...)
        

        if do_generate_article:
            if information_table is None: 
                同上 load_information_table_from_local_fs
            if outline is None:
                outline = self._load_outline_from_local_fs(topic...)
            draft_article: StormArticle 
                = self.run_article_generation_module(outline, 
                                                     information_table,...)

        if do_polish_article:
            if draft_article is None:
                draft_article = self._load_draft_article_from_local_fs(topic,...)
            self.run_article_polishing_module(draft_article, remove_duplicate)
    
    runner. post_run()
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(config_log, 檔案位址 )

        llm_call_history :list  = self.lm_configs.collect_and_reset_lm_history()

        with open( ...'llm_call_history.jsonl', 'w') as f:
            for call in llm_call_history:
                if 'kwargs' in call:
                    call.pop('kwargs')  # All kwargs are dumped together to run_config.json.
                f.write(json.dumps(call) + '\n')

    runner.summary()
    # class Engine
        print("***** Execution time *****")
        for k, v in self.time.items():
            print(f"{k}: {v:.4f} seconds")

        print("***** Token usage of language models: *****")
        for k, v in self.lm_cost.items():
            print(f"{k}")
            for model_name, tokens in v.items():
                print(f"    {model_name}: {tokens}")

        print("***** Number of queries of retrieval models: *****")
        for k, v in self.rm_cost.items():
            print(f"{k}: {v}")



