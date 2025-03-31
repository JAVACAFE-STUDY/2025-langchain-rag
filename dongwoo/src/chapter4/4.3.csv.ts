// https://js.langchain.com/docs/integrations/document_loaders/file_loaders/csv/
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";

const exampleCsvPath = "/Users/user/Downloads/Sample Data v3/OrderAllocation_v2.csv";

const loader = new CSVLoader(exampleCsvPath);

const docs = await loader.load();
console.log(docs[0]);