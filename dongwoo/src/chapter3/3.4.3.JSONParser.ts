/**
 * JSON 파서를 사용한 예제
 */
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from '@langchain/core/prompts';

import { JsonOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';
import {z} from 'zod';

import 'dotenv/config';

// 미리 정의된 JSON 스키마
type Answer = {
    continent: string;
    population: string;
};

// pydantic 이랑 비슷한 방식이 있을 것 같은데
const formatInstructions = `
{ 
    continent: string           // continent는 사용자 물어본 나라가 속한 대륙이야
    , population: string        // 사용자가 물어본 나라의 인구인데 숫자형태로 부탁해. 3자리별로 ,도 부탁해
}
`;

// https://zod.dev/ --> 정적 타입 추론을 지원하는 TypeScript 기반 스키마 검증 라이브러리
// 

const AnswerSchema = z.object({
    continent: z.string().describe("continent는 사용자 물어본 나라가 속한 대륙이야"),
    population: z.string().describe("사용자가 물어본 나라의 인구인데 숫자형태로 부탁해. 3자리별로 ,도 부탁해"),
});

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
// const outputParser = new JsonOutputParser<Answer>();
const outputParser = StructuredOutputParser.fromZodSchema(AnswerSchema);

// const prompt = await PromptTemplate.fromTemplate('Answer the user query.\n{formatInstructions}\n{query}')
//     .partial({formatInstructions: formatInstructions});

const prompt = await PromptTemplate.fromTemplate('Answer the user query.\n{formatInstructions}\n{query}')
    .partial({formatInstructions: outputParser.getFormatInstructions()});

const chain2 = prompt.pipe(model).pipe(outputParser);

const result = await chain2.invoke({query: "대한민국은 어떤나라야?"})

console.log(result);



