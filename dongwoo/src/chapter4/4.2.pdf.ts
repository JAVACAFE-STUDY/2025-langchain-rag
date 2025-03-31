// https://js.langchain.com/docs/how_to/document_loader_pdf/
// https://v03.api.js.langchain.com/classes/_langchain_community.document_loaders_fs_pdf.PDFLoader.html
// OCR 쓰는 건 어렵네.
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

// const loader = new PDFLoader("/Users/user/Downloads/2024_리그규정.pdf");
const loader = new PDFLoader("/Users/user/Downloads/2024_연감.pdf", {
    
});

const docs = await loader.load();

console.log(docs[508].pageContent);
console.log(docs[508].metadata);


