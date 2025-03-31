// https://js.langchain.com/docs/how_to/recursive_text_splitter/

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

const texts = await splitter.splitDocuments(docs);

console.log("texts.length", texts.length);

for(const text of texts) {
    process.stdout.write(text.pageContent.length + " ");
}


