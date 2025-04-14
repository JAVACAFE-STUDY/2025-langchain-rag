import {ChatPromptTemplate} from  '@langchain/core/prompts';
import {ChatOpenAI} from '@langchain/openai';
import 'dotenv/config';

const prompt = ChatPromptTemplate.fromTemplate('{topic}에 대한 가벼운 조크를 이야기해줘.');
const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
});

const chain = prompt.pipe(model);

const stream = await chain.stream({"topic": "아이스크림"});

for await (const chunk of stream) {
    process.stdout.write(chunk.content as string);
}