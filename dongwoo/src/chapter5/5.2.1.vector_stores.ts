import {ChromaClient} from 'chromadb';
import {OpenAIEmbeddingFunction} from 'chromadb';
import 'dotenv/config'
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const loader = new PDFLoader("/Users/user/Downloads/대한민국헌법(헌법)(제00010호)(19880225).pdf");
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
});

const texts = await splitter.splitDocuments(docs);

const client = new ChromaClient();
const embeddingFunction = new OpenAIEmbeddingFunction({
    openai_api_key: process.env.OPENAI_API_KEY,
    openai_model: 'text-embedding-3-small'
});

// await client.deleteCollection({
//     name: "my_collection",
// });


const collection = await client.getOrCreateCollection({
    name: "my_collection",
    embeddingFunction: embeddingFunction,
});

  // 문서 데이터 준비

// const documents = [
//     "This is the first document.",
//     "Here is another piece of text.",
//     "This document discusses JavaScript and embeddings.",
// ];
// const metadatas = [
//     { source: "doc1" },
//     { source: "doc2" },
//     { source: "doc3" },
// ];
// const ids = ["id1", "id2", "id3"];  
const documents = texts.map((text) => text.pageContent);
const metadatas = texts.map((text) => ({source: text.metadata.source, page: text.metadata.loc.paggeNumber}));
const ids = Array.from({length:texts.length}, (v,i)=>'' + (i+1));

// console.log("documents.length", documents.length, documents);
// console.log("metadatas.length", metadatas.length, metadatas);
// console.log("ids.length", ids.length, ids);

// type Metadata = Record<string, string | number | boolean>;

await collection.add({
    ids,
    documents,
    metadatas,
});

  // 쿼리 수행
const queryResults = await collection.query({
    queryTexts: ["대통령의 임기는?"],
    nResults: 2, // 반환할 결과 개수
});

console.log("Query Results:", queryResults);

