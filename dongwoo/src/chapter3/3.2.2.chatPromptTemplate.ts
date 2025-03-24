/**
 * ChatPromptTemplate 샘플입니다.
 * 
 */
import {ChatPromptTemplate} from '@langchain/core/prompts';
import {ChatOpenAI} from '@langchain/openai';
import {SystemMessage, HumanMessage, AIMessage} from '@langchain/core/messages';

import 'dotenv/config';

const promptTemplate = ChatPromptTemplate.fromMessages([
    new SystemMessage("You are a helpful AI bot. Your name is {name}."),
    new HumanMessage("Hello! how are you doing?"),
    new AIMessage("I'm doing well, thank you."),
    new HumanMessage("{user_input}")
]);

// format을 활용해서 위 매개 변수에 값을 할당한다.
const prompt = await promptTemplate.format({name: "Bob", user_input: "What is your name?"});

console.log(prompt);

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

const result = await model.invoke(prompt);

console.log(result.content);





