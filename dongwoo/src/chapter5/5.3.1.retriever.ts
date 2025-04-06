import { Chroma } from "@langchain/community/vectorstores/chroma";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from '@langchain/openai';

import 'dotenv/config'

const loader = new PDFLoader("/Users/user/Downloads/대한민국헌법(헌법)(제00010호)(19880225).pdf");
const pages = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
});

const texts = await splitter.splitDocuments(pages);
const changedText = texts.map(text => ({
    ...text,
    metadata: {source: text.metadata.source, page: text.metadata.loc.paggeNumber}
}));

const client = await Chroma.fromDocuments(changedText, new OpenAIEmbeddings({
    model: 'text-embedding-3-small'
}), {
    collectionName: 'my_collection2',
    collectionMetadata: {
        "hnsw:space": "cosine",
    }
});


const retriever = client.asRetriever({
    // searchType: 'mmr',
    k: 3,
    // searchKwargs: {
    //     "fetchK": 10,
    //     lambda: 0.5,
    // }
});

const result = await retriever.invoke('국회의원의 의무');
// const result = await retriever.getRelevantDocuments('국회의원의 의무');

console.log(result);

// https://github.com/langchain-ai/langchain/discussions/18125
/*
Chroma에서 Maximal Marginal Relevance (MMR) 검색과 관련된 기능이 제한되거나 지원되지 않는 이유는, 주로 최근의 기술적 변경 사항과 구현상의 차이에 기인합니다.

### **MMR 검색 지원 중단 시점**
Chroma에서 `similaritySearchWithRelevanceScores`와 같은 기능이 MMR 검색과 함께 작동하지 않는다는 논의는 2024년 초부터 나타났습니다. 예를 들어, 2024년 2월 27일 기준으로 LangChain의 GitHub 논의에서 Chroma가 MMR 검색 중 점수를 반환하지 못한다는 언급이 있었습니다[1]. 이는 Chroma가 기본적으로 MMR 알고리즘을 지원하지만, 점수 반환과 같은 고급 기능은 구현되지 않았거나 제한적으로 제공된다는 것을 의미합니다.

### **왜 지원이 제한되었는가?**
1. **기능 구현의 우선순위**  
   Chroma와 LangChain은 오픈소스 프로젝트로, 새로운 기능 추가나 기존 기능 개선은 커뮤니티의 요구와 프로젝트 리소스에 따라 결정됩니다. MMR 검색에서 점수를 반환하는 기능은 일부 사용 사례에만 필요할 수 있으므로 우선순위에서 밀렸을 가능성이 있습니다.

2. **기술적 문제**  
   MMR 알고리즘은 결과의 다양성과 유사성을 동시에 최적화하기 위해 설계되었습니다. 그러나 이를 구현하는 과정에서 점수를 반환하는 로직이 복잡해질 수 있으며, Chroma의 기존 데이터 구조나 API 설계와 충돌했을 가능성이 있습니다.

3. **오류 및 버그**  
   GitHub 이슈에서 언급된 바와 같이, 코드 내에서 `lambda_mult`와 같은 매개변수 처리에 오류가 있었던 것으로 보입니다[2]. 이러한 문제는 기능이 제대로 작동하지 않게 만들었을 수 있습니다.

4. **대체 메서드 제공**  
   Chroma는 여전히 기본적인 MMR 검색을 지원하며, 사용자는 이를 통해 다양성과 유사성을 최적화할 수 있습니다[3][5]. 그러나 점수 반환 기능이 필요한 경우 Python SDK를 수정하거나 다른 벡터 데이터베이스를 사용하는 것이 필요할 수 있습니다.

### **결론**
Chroma에서 `similaritySearchWithRelevanceScores`와 같은 기능이 MMR 검색과 함께 작동하지 않는 것은 2024년 초부터 명확히 드러났으며, 이는 기술적 제한과 우선순위의 문제로 보입니다. 현재로서는 Python SDK를 수정하거나 다른 벡터 데이터베이스를 활용하는 것이 이러한 제한을 극복하는 방법입니다.

Citations:
[1] https://github.com/langchain-ai/langchain/discussions/18229
[2] https://github.com/hwchase17/langchain/issues/4861
[3] https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html
[4] https://www.restack.io/p/similarity-search-answer-chroma-cat-ai
[5] https://api.python.langchain.com/en/latest/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
[6] https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/chroma.py
[7] https://www.kaggle.com/code/youssef19/information-retrieval-with-langchain
[8] https://lagnchain.readthedocs.io/en/stable/_modules/langchain/vectorstores/chroma.html
[9] https://v03.api.js.langchain.com/functions/_langchain_core.utils_math.maximalMarginalRelevance.html
[10] https://lagnchain.readthedocs.io/en/stable/modules/prompts/example_selectors/examples/mmr.html
[11] https://www.reddit.com/r/LangChain/comments/17r908t/how_to_add_ids_in_chroma_using_langchain/
[12] https://wikidocs.net/231585
[13] https://rudaks.tistory.com/entry/langchain-%EB%B2%A1%ED%84%B0%EC%A0%80%EC%9E%A5%EC%86%8C-%EC%A7%80%EC%9B%90-%EA%B2%80%EC%83%89%EA%B8%B0VectorStore-backed-Retriever
[14] https://mobicon.tistory.com/616
[15] https://wikidocs.net/234009
[16] https://api.python.langchain.com/en/latest/_modules/langchain_community/vectorstores/neo4j_vector.html
[17] https://stackoverflow.com/questions/76962785/how-to-check-for-duplicate-documents-in-vectorstore-efficiently
[18] https://www.onsemi.com/pub/collateral/hbd853-d.pdf
[19] https://github.com/langchain-ai/langchain/issues/11815
[20] https://arxiv.org/html/2411.11895v1
[21] https://python.langchain.com/v0.2/docs/tutorials/rag/
[22] https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/
[23] https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoMMR/
[24] https://www.linkedin.com/pulse/langchain-retrieval-rutam-bhagat-d7kzf
[25] https://huggingface.co/spaces/SorbonneUniversity/SorboBot/commit/fc4094134671566052cc25bc287c089340bfcec6
[26] https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.maximal_marginal_relevance.html
[27] https://api.python.langchain.com/en/latest/chroma/vectorstores/langchain_chroma.vectorstores.maximal_marginal_relevance.html

---
Perplexity로부터의 답변: https://www.perplexity.ai/search/similaritysearchwithrelevencas-.AypmW8bTlWjHwaP6yx6cQ?utm_source=copy_output
*/
const result2 = await client.similaritySearchWithScore('국회의원의 의무');

console.log(result2[0][0].pageContent);
console.log(result2[0][1]);

