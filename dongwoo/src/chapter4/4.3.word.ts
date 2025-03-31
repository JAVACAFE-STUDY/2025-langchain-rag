// https://js.langchain.com/docs/integrations/document_loaders/file_loaders/docx/
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";

// const loader = new PDFLoader("/Users/user/Downloads/2024_리그규정.pdf");
const loader = new DocxLoader("/Users/user/Downloads/file-sample_1MB.docx");

const docs = await loader.load();

console.log(docs);


