/**
 * Models의 예제 중에서 앤트로픽의 Claude-3를 사용하는 예제입니다.
 * 설치: pnpm add @langchain/anthropic @langchain/core
 * 
 * 랭채인의 Models를 활용할 경우 기존의 다른 LLM의 경우는 각각의 LLM 사용법에 대해서 익혀야 하지만
 * 랭체인의 Models의 경우는 동일한 인터페이스를 제공하기 때문에 각각의 LLM에 대한 사용법을 익히면 다른 LLM을 사용하는 것도 어렵지 않습니다.
 * 
 * 프롬프트의 3가지 형태
 * 1. SystemMessage: LLM에게 역할을 부여하는 메시지
 * 2. HumanMessage: LLM에게 전달하는 사용자의 메시지
 * 3. AIMessage: LLM이 출력한 메시지
 *    3.1. AIMessage의 경우는 API 호출 결과로 받아보는 '현재대화'
 *    3.2. 히스토리로 넣어주는 '과거 대화'
 *         쳇봇 시스템을 구축할 경우 앞서 진행된 대화의 히스토리를 LLM에게 전달하는 것이 중요한데, 여기서 AI가 대답한 부분을 AIMessage로 넣어주면 됩니다.
 * 
 */
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

import 'dotenv/config'

const model = new ChatAnthropic({
    model: "claude-3-haiku-20240307",
    temperature: 0
});

const messages = [
    new SystemMessage("Translate the following from English into Korean"),
    new HumanMessage("hi!"),
];

const result = await model.invoke(messages);
console.log(result.content);
