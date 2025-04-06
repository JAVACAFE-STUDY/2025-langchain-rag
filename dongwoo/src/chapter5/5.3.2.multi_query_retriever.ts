import { Chroma } from "@langchain/community/vectorstores/chroma";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";

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

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });
const retrieverFromLLM = MultiQueryRetriever.fromLLM({
    retriever: client.asRetriever(),
    llm
});


const result = await retrieverFromLLM.invoke('국회의원의 의무는 무엇인가요?');

console.log(result);



