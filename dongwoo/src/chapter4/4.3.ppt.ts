// https://js.langchain.com/docs/integrations/document_loaders/file_loaders/pptx/
import { PPTXLoader } from "@langchain/community/document_loaders/fs/pptx";

const loader = new PPTXLoader("/Users/user/Downloads/Copilot-scenarios-for-Marketing.pptx");

const docs = await loader.load();

console.log(docs[0]);