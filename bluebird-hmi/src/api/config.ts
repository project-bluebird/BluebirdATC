import type { ConfigFile } from "@rtk-query/codegen-openapi";

const config: ConfigFile = {
  schemaFile: "http://127.0.0.1:8000/openapi.json",
  apiFile: "./emptyApi.ts",
  apiImport: "emptySplitApi",
  outputFile: "api.ts",
  exportName: "api",
  hooks: true,
};

export default config;
