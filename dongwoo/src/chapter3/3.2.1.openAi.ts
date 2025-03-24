/**
 * Models의 예제 중에서 OpenAI를 사용하는 예제입니다.
 * 설치: pnpm add @langchain/openai @langchain/core
 * 실행: pnpm exec tsx src/chapter3/3.2.openAi.ts
 */

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import 'dotenv/config'

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

const messages = [
    new SystemMessage("Translate the following from English into Korean"),
    new HumanMessage("hi!"),
];

const result = await model.invoke(messages);
console.log(result.content);
