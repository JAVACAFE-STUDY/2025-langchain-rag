import {ChatPromptTemplate} from  '@langchain/core/prompts';
import {ChatOpenAI} from '@langchain/openai';
import 'dotenv/config';

const prompt = ChatPromptTemplate.fromTemplate('다음 한글 문장을 프랑스어로 번역해줘. {sentence}');
const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
});


const chain = await prompt.pipe(model);

const result = await chain.batch([
    {"sentence": "안녕하세요!"},
    {"sentence": "오늘 날씨가 참 좋네요."},
    {"sentence": "저녁에 친구들과 영화를 볼 거에요."},
    {"sentence": "그 학생은 매우 성실하게 공부해요."},
    {"sentence": "커피 한 잔이 지금 딱 필요해요."},
]);


result.forEach((response, index) => {
    console.log(`Response ${index + 1}: ${response.content}`);
}
);

