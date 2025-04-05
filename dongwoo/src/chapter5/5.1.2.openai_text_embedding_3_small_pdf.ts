import { OpenAIEmbeddings } from "@langchain/openai";

import 'dotenv/config'

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";


const loader = new PDFLoader("/Users/user/Downloads/2024_연감.pdf");

const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    separators: ["\n", '\n\n', ' ', ''],
    chunkSize: 500,
    chunkOverlap: 100,
    lengthFunction: (text) => text.length,
});

const texts = (await splitter.splitDocuments(docs)).map((text) => text.pageContent);

const embeddingModel = new OpenAIEmbeddings({model: 'text-embedding-3-small'});
const embeddings = await embeddingModel.embedDocuments(texts);

console.log(embeddings.length, embeddings[0].length);


