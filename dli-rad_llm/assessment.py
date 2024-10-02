####################################################################################
## TODO: Your workspace is below
    
llama_full_prompt = PromptTemplate.from_template(template="<s>[INST]<<SYS>>{sys_msg}<</SYS>>\n\nContext:\n{history}\n\nHuman: {input}\n[/INST] {primer}",)

llama_prompt = llama_full_prompt.partial(
    sys_msg = ( 
        "You are a helpful, respectful and honest AI assistant."
        "\nAlways answer as helpfully as possible, while being safe."
        "\nPlease be brief and efficient unless asked to elaborate, and follow the conversation flow."
        "\nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        "\nEnsure that your responses are socially unbiased and positive in nature."
        "\nIf a question does not make sense or is not factually coherent, explain why instead of answering something incorrect." 
        "\nIf you don't know the answer to a question, please don't share false information."
        "\nIf the user asks for a format to output, please follow it as closely as possible."
    ),
    primer = "",
    history = "",
)
llama_hist_prompt = llama_prompt.copy()
llama_hist_prompt.input_variables = ['input', 'history']

####################################################################################
## THESE MIGHT BE USEFUL IMPORTS!

from langchain.chains import ConversationChain
#from langchain.memory import ConversationSummaryMemory

img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
emo_pipe = pipeline('sentiment-analysis', 'SamLowe/roberta-base-go_emotions')  
zsc_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
tox_pipe = pipeline("text-classification", model="nicholasKluge/ToxicityModel")
## WARNING: toxic_pipe returns the reward, where reward = 1 - toxicity

###################################################################################


class MyAgent(MyAgentBase):
    
    ## Instance methods that can be passed in as BaseModel arguments. 
    ## Will be associated with self
    
    general_prompt : PromptTemplate
    llm            : BaseLLM
    
    general_chain  : Optional[LLMChain]
    max_messages   : int                   = Field(10, gt=1)
    
    temperature    : float                 = Field(0.6, gt=0, le=1)
    max_new_tokens : int                   = Field(128, ge=1, le=2048)
    eos_token_id   : Union[int, List[int]] = Field(2, ge=0)
    gen_kw_keys = ['temperature', 'max_new_tokens', 'eos_token_id']
    gen_kw = {}
    
    user_toxicity  : float = 0.5
    user_emotion   : str = "Unknown"
    
    
    @root_validator
    def validate_input(cls, values: Any) -> Any:
        '''Think of this like the BaseModel's __init__ method'''
        if not values.get('general_chain'):
            llm = values.get('llm')
            prompt = values.get("general_prompt")
            values['general_chain'] = LLMChain(llm=llm, prompt=prompt)  ## <- Feature stop 
        values['gen_kw'] = {k:v for k,v in values.items() if k in values.get('gen_kw_keys')}
        return values
    

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any): 
        '''Takes in previous logic and generates the next action to take!'''
        
        ## [Base Case] Default message to start off the loop. TO NOT OVERRIDE
        tool, response = "Ask-For-Input Tool", "Hello World! How can I help you?"
        if len(intermediate_steps) == 0:
            return self.action(tool, response)
        
        ## History of past agent queries/observations
        queries      = [step[0].tool_input for step in intermediate_steps]
        observations = [step[1]            for step in intermediate_steps]
        last_obs     = observations[-1]    # Most recent observation (i.e. user input)

        #############################################################################
        ## FOR THIS METHOD, ONLY MODIFY THE ENCLOSED REGION
        
        ## [!] Probably a good spot for your user statistics tracking
        #if emo_pipe(str(last_obs))[0].get("score") > 0.6:
        #user_emotion = emo_pipe(str(queries))[0].get("label")
        self.user_emotion = emo_pipe(str(last_obs))[0].get("label")
        self.user_toxicity = 1 - tox_pipe(str(last_obs))[0].get("score")
        #print(self.user_emotion)
        #if tox_pipe(str(last_obs))[0].get("score") > 0.6:
        #print("UE:"+user_emotion)
        ## [Stop Case] If the conversation is getting too long, wrap it up
        if len(observations) >= self.max_messages:
            response = "Thanks so much for the chat, and hope to see ya later! Goodbye!"
            return self.action(tool, response, finish=True)
        
        ## [!] Probably a good spot for your input-augmentation steps
        if last_obs.find("```"):
            last_obs = last_obs.replace(' ```', '')
        with SetParams(llm, **self.gen_kw):
            response = self.general_chain.run(last_obs)
            
        ## [!] Probably a good spot for your output-postprocessing steps
        ## [Default Case] If observation is provided and you want to respond... do it!
        ## FOR THIS METHOD, ONLY MODIFY THE ENCLOSED REGION
        #############################################################################
        
        ## [Default Case] Send over the response back to the user and get their input!
        i3 = last_obs.find("img-files")
        if i3 > 0:
            array_with_image_path = last_obs.split("`")
            for image_path in array_with_image_path:
                i4 = image_path.find("img-files")
                if i4 >= 0:
                    response = img_pipe(image_path)[0].get("generated_text")
                    return self.action(tool, response)
                
        i1 = response.find("def")
        if i1 > 0:
            text = response
            i2 = text.find("```", i1)
            response = text[i1:i2-1]+"\n"
        return self.action(tool, response)
    
    
    def reset(self):
        self.user_toxicity = 0
        self.user_emotion = "Unknown"
        if getattr(self.general_chain, 'memory', None) is not None:
            self.general_chain.memory.clear()  ## Hint about what general_chain should be...


####################################################################################
## Define how you want your conversation to go. You can also use your own input
## The below example in conversation_gen exercises some of the requirements.

student_name = "John Doe"   ## TODO: What's your name
ask_via_input = False       ## TODO: When you're happy, try supplying your own inputs

def conversation_gen():
    #yield f"Hello! How's it going? My name is {student_name}! Nice to meet you!"
    yield f"Hello! How are you doing today? My name is Amy, and I'm a punk rock sensation from the '80s!"
    #yield "Please tell me a little about deep learning!"
    #yield "What's my name?"                                  ## Memory buffer
    #yield "I'm not feeling very good -_-. What should I do"  ## Emotion sensor
    #yield "No, I'm done talking! Thanks so much!"            ## Conversation ender
    #yield "Goodbye!"                                         ## Conversation ender x2
    #yield "Please provide a basic fibonacci application with ``` python!"
    yield "Can you implement a quicksort method with ``` python?!"
    #yield "Ok! I'm looking at this image, and I need some help. Can you describe the image `img-files/.5453435.jpg`"
    raise KeyboardInterrupt()

conversation_instance = conversation_gen()
converser = lambda x: next(conversation_instance)

if ask_via_input:
    converser = input  ## Alternatively, supply your own inputs

conv_chain = ConversationChain(llm=llm, prompt=llama_hist_prompt, verbose=True)
agent_kw = dict(
    llm = llm,
    general_prompt = llama_hist_prompt,
    general_chain = conv_chain,
    temperature = 0.6,
    max_new_tokens = 128,
    eos_token_id = [2]
)
#print("emo_pipe:"+str(emo_pipe(conv_chain.memory)))
#print("tox_pipe:"+str(zsc_pipe(tox_pipe(str(conv_chain.memory)))))
#print("tox_pipe:"+str(tox_pipe(str(conv_chain.memory))[0]))
#print("emo_pipe:"+str(emo_pipe(str(conv_chain.memory))[0].get("label")))
agent_ex = AgentExecutor.from_agent_and_tools(
    agent = MyAgent(**agent_kw),
    tools=[AskForInputTool(converser).get_tool()], 
    verbose=True
)

## NOTE: You might want to comment this out to make testing the autograder easier
try: agent_ex.run("")
except KeyboardInterrupt: print("KeyboardInterrupt")