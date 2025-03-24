/**
 * 퓨샷(Few-shot): 딥러닝 모델이 결과물을 출력할 때 예시 결과물을 제시함으로써 원하는 결과물로 유도하는 방법론
 */
import {PromptTemplate, FewShotPromptTemplate} from '@langchain/core/prompts';
import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config';

// 예시 결과물
const examples = [{
    question: '아이유로 삼행시 만들어줘',
    answer: `
        아: 아이유는
        이: 이런 강의를 들을 이
        유: 유가 없다.
    `,
}];

const examplePrompt = PromptTemplate.fromTemplate("Question: {question}\nAnswer: {answer}");

// console.log(await examplePrompt.format(examples[0]));

const fewPrompt = new FewShotPromptTemplate({
    examples,
    examplePrompt,
    suffix: "Question: {input}",
    inputVariables: ["input"]
});

// console.log(await fewPrompt.format({input: "호날두로 삼행시 만들어줘"}));

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

// const result = await model.invoke("호날두로 삼행시 만들어줘");

// 결과를 구한다.
const result = await model.invoke(await fewPrompt.format({input: "호날두로 삼행시 만들어줘"}));

console.log(result.content);