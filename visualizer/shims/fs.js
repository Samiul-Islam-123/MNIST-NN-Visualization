// Browser shim for Node's "fs" — lets the real NN/Network.js load in the browser.
// Network.js only touches fs in save()/load(), which the browser never calls
// (models arrive via JSON upload / socket payloads instead).
const unavailable = (name) => () => {
  throw new Error(`fs.${name} is not available in the browser`);
};

export default {
  existsSync: () => false,
  readFileSync: unavailable("readFileSync"),
  writeFileSync: unavailable("writeFileSync"),
  promises: {
    readFile: unavailable("promises.readFile"),
    writeFile: unavailable("promises.writeFile")
  }
};
