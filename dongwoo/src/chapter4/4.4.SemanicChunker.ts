import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import 'dotenv/config'

async function createSemanticChunks(text: string): Promise<string[]> {
    // OpenAI 임베딩 초기화
    const embeddings = new OpenAIEmbeddings();

    // 텍스트를 문장 단위로 분리
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1024, // 각 청크의 최대 크기
        chunkOverlap: 128, // 청크 간 겹치는 부분
    });

    const splitTexts = await textSplitter.splitText(text);

    // 각 문장의 임베딩 생성
    const embeddingsList = await Promise.all(
        splitTexts.map(async (chunk) => await embeddings.embedQuery(chunk))
    );

    // 코사인 유사도를 계산하여 의미론적 청크 생성 (예: 임계값 기반)
    const semanticChunks = [];
    let currentChunk = [splitTexts[0]];

    for (let i = 1; i < splitTexts.length; i++) {
        const similarity = calculateCosineSimilarity(
            embeddingsList[i - 1],
            embeddingsList[i]
        );

        if (similarity < 0.8) { // 임계값 설정
            semanticChunks.push(currentChunk.join(" "));
            currentChunk = [];
        }
        currentChunk.push(splitTexts[i]);
    }

    if (currentChunk.length > 0) {
        semanticChunks.push(currentChunk.join(" "));
    }

    return semanticChunks;
}

// 코사인 유사도 계산 함수
function calculateCosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// 테스트 실행
(async () => {
    const text = "This is a sample text. It contains multiple sentences about different topics.";
    const chunks = await createSemanticChunks(text);
    console.log(chunks);
})();


