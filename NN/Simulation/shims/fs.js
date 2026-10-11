// Browser shim for Node's "fs" module to allow Network.js to load in browser ES modules
export default {
  existsSync: () => false,
  readFileSync: () => { throw new Error("fs.readFileSync is not available in browser"); },
  writeFileSync: () => { throw new Error("fs.writeFileSync is not available in browser"); }
};
