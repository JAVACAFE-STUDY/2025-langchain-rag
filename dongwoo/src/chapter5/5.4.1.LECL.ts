import {StringOutputParser} from '@langchain/core/output_parsers';
import {ChatPromptTemplate} from  '@langchain/core/prompts';
import {ChatOpenAI} from '@langchain/openai';
import 'dotenv/config';

const prompt = ChatPromptTemplate.fromTemplate('{topic}에 대한 가벼운 조크를 이야기해줘.');
const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
});

const outputParser = new StringOutputParser();

// LECEL 체인 생성 prompt | model | outputParser
const chain = prompt.pipe(model).pipe(outputParser);

// 체인을 실행
const response = await chain.invoke({
    topic: '아이스크림',
});

console.log(response);
