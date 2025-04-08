/**
 * 문서 백터를 재가공하여 검색 품질을 향상
 */

import {MultiVectorRetriever} from 'langchain/retrievers/multi_vector';
import { InMemoryStore } from "@langchain/core/stores";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from '@langchain/openai';
import { v4 as uuidv4 } from "uuid"; 

import 'dotenv/config'
import { Document } from '@langchain/core/documents';

const loader = new PDFLoader("/Users/user/Downloads/대한민국헌법(헌법)(제00010호)(19880225).pdf");
const pages = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
});

const docs = await splitter.splitDocuments(pages);
const flatDocs = docs.map(text => ({
    ...text,
    metadata: {source: text.metadata.source, page: text.metadata.loc.paggeNumber}
}));

const client = await Chroma.fromDocuments(flatDocs, new OpenAIEmbeddings({
    model: 'text-embedding-3-small'
}), {
    collectionName: 'my_collection2',
    collectionMetadata: {
        "hnsw:space": "cosine",
    }
});

const store = new InMemoryStore();
const idKey = 'doc_id';

const retriever = new MultiVectorRetriever({
    vectorstore: client,
    idKey,
    byteStore: store,
});

const docIds = pages.map(() => uuidv4());

const childSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,
    chunkOverlap: 0,
});

const subDocs = [];

for (let i = 0; i < flatDocs.length; i += 1) {
    const childDocs = await childSplitter.splitDocuments([flatDocs[i]]);
    const taggedChildDocs = childDocs.map((childDoc) => {
      // eslint-disable-next-line no-param-reassign
      childDoc.metadata[idKey] = docIds[i];
      return childDoc;
    });
    subDocs.push(...taggedChildDocs);
}

await retriever.vectorstore.addDocuments(subDocs);

const keyValuePairs: [string, Document][] = flatDocs.map((originalDoc, i) => [
    docIds[i],
    originalDoc,
]);

await retriever.docstore.mset(keyValuePairs);

console.log('[하위 청크] \n');
const result = await retriever.vectorstore.similaritySearch('국민의 권리');
console.log(result[0].pageContent);

console.log('[상위 청크] \n');

const result2 = await retriever.invoke('국민의 의무');
console.log(result2[0].pageContent);


