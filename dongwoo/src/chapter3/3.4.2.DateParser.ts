/**
 * 날짜 형식만 출력하는 DateTimeOutputParser 예제
 */
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';

// @langchain/core/output_parsers 에서 존재하지 않는다.
// 두개를 비교해봐야 하나? 버전이 없하면 또 변경될 수 있을 것 같다.
import { DatetimeOutputParser } from 'langchain/output_parsers';

import 'dotenv/config';


const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const outputParser = new DatetimeOutputParser();

const template = `
    Ansewer the users question:
    {question}

    {formatInstructions}
`;

const prompt = PromptTemplate.fromTemplate(template);

const chain = await prompt.partial({formatInstructions: outputParser.getFormatInstructions()});
const chain2 = chain.pipe(model);

const result = await chain2.invoke({question: "비트코인은 언제 개발됐어?"})

console.log(result.content);



