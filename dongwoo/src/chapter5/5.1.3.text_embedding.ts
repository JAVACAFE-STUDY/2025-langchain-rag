import { OpenAIEmbeddings } from "@langchain/openai";
import 'dotenv/config'
import { dot, norm  } from 'mathjs';

const embeddingModel = new OpenAIEmbeddings({model: 'text-embedding-3-small'});
const examples = await embeddingModel.embedDocuments([
    '안녕하세요.',
    '제 이름은 홍두깨입니다.',
    `이름이 무엇인가요?`,
    `랭체인은 유용합니다.`
]);

const embeded_query_q = await embeddingModel.embedQuery('이 대화에서 언급된 이름은 무엇인가요?');
const embeded_query_a = await embeddingModel.embedQuery('이 대화에서 언급된 이름은 홍길동입니다.');

function cosSim(a: number[], b: number[]): number {
    // 내적 계산
    const dotProduct = dot(a, b) as number;

    // 벡터의 노름 계산
    const normA = norm(a) as number;
    const normB = norm(b) as number;

    // 코사인 유사도 계산
    return dotProduct / (normA * normB);    
}

console.log(cosSim(embeded_query_q, embeded_query_a));
console.log(cosSim(embeded_query_a, examples[1]));
console.log(cosSim(embeded_query_q, examples[3]));




