/**
 * 실습하기
 */
import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage } from "@langchain/core/messages";
import {ChatPromptTemplate, HumanMessagePromptTemplate} from '@langchain/core/prompts';
import 'dotenv/config';

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0,   });

const promptTemplate = ChatPromptTemplate.fromMessages([
    new SystemMessage("당신은 공부 계획을 세워주는 스터디 플래너 머신입니다."),
    new SystemMessage("사용자의 공부 주제를 입력받으면, 이를 학습하기 위한 공부 계획을 작성합니다."),
    HumanMessagePromptTemplate.fromTemplate("{text}"),
]);

const messages = await promptTemplate.formatMessages({text: "수학 공부 계획을 세워줘."});
const stream = await model.stream(messages);

for await (const chunk of stream) {
    process.stdout.write(chunk.content as string);
}


