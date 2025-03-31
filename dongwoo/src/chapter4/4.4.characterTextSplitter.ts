import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from '@langchain/textsplitters';


const loader = new PDFLoader("/Users/user/Downloads/2024_연감.pdf");

const docs = await loader.load();

const splitter = new CharacterTextSplitter({
    separator: "\n",
    chunkSize: 500,
    chunkOverlap: 100,
    lengthFunction: (text) => text.length,
});

const texts = await splitter.splitDocuments(docs);

for(const text of texts) {
    process.stdout.write(text.pageContent.length + " ");
}





