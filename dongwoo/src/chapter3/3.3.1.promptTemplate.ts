/**
 * PromptTemplate의 역할은 사용자 입력을 활용하여 원하는 형태로 가공하는 것
 * 
 * 종류
 * PromptTemplate: 기본적으로 사용자의 입력을 가공
 * ChatPromptTemplate: 시스템 메세지를 설정하여 LLM 역할을 부여할 수 있는 템플릿
 */
import { ChatOpenAI } from "@langchain/openai";
import {PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts';
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";
import {LLMChain} from 'langchain/chains'; 
import 'dotenv/config';

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

// 공통적인 역할을 미리 만들어놓는다.
// const prompt = PromptTemplate.fromTemplate(`
//     너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 {개수}추천하고,
//     그 요리의 래시피를 제시해줘. 내가 가진 재료는 아래와 같아.
//     <재료>
//     {재료}    
// `);

// // 실제 입력은 이런식으로 받는다.
// const messages = await prompt.format({
//     개수: 3,
//     재료: [
//         "닭고기",
//         "당근",
//         "양파",
//         "감자",
//         "소금",
//         "후추",
//         "올리브유",
//     ],
// });

// console.log(messages);


// const stream = await model.stream(messages);

// for await (const chunk of stream) {
//     process.stdout.write(chunk.content as string);
// }

// ChatPromptTemplate은 PromptTemplate에서 SystemMessage, HumanMessage, AIMessage 가 추가됩니다.
// 이미 나눈 대화를 넣어 맥락을 더해줄 수 있습니다.
// ChatPromptTemplate은 챗봇에 역할 부여 및 대화에 맥락을 추가하는 부분에 대해서 기본적인 PromptTemplate 보다 구분되어서 적용이 가능하다.

const systemMessage = new SystemMessage("너는 항상 밝은 말투로 대화하는 챗봇이야. 답변의 끝에 이모티콘을 붙여줘.");
const humanMessage = new HumanMessage("오늘은 날씨가 어때?");
const aiMessage = new AIMessage("오늘은 날씨가 맑고 화창해요! 😊");
const humanMessage2 = new HumanMessage("{input}");



const prompt2 = await ChatPromptTemplate
    .fromMessages([systemMessage, humanMessage, aiMessage])

// 체인: LLM과 다양한 모듈을 하나의 패키지로 묶어주는 모듈
// ChatPromptTemplate 에 ChatOpenAI 모듈을 붙이는 예시
// 체인을 설정하는 방법이 책과 다름(책의 방식은 Deprecated 됨)
const chain = prompt2.pipe(model);

const stream2 = await chain.stream("오늘 너의 기분은 어때?");

for await (const chunk of stream2) {
    process.stdout.write(chunk.content as string);
}
