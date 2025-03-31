// https://js.langchain.com/docs/integrations/document_loaders/web_loaders/web_puppeteer/

import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";

const loader = new PuppeteerWebBaseLoader("https://www.naver.com", {
  // required params = ...
  // optional params = ...
});

const docs = await loader.load();
console.log(docs[0]);