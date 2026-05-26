const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath, files);
    } else {
      files.push(fullPath);
    }
  }
  return files;
}

const files = walk(root);

for (const file of files.filter((item) => item.endsWith('.json'))) {
  JSON.parse(fs.readFileSync(file, 'utf8'));
}

for (const file of files.filter((item) => item.endsWith('.js') && !item.endsWith('check_static.js'))) {
  new Function(fs.readFileSync(file, 'utf8'));
}

const mojibakePatterns = ['浣', '鑸', '闈', '缁', '鍥', '璇', '鏌', '鐨', '鏂'];
const offenders = [];
for (const file of files.filter((item) => /\.(js|wxml|wxss|json)$/.test(item) && !item.includes(`${path.sep}tools${path.sep}`))) {
  const text = fs.readFileSync(file, 'utf8');
  if (mojibakePatterns.some((pattern) => text.includes(pattern))) {
    offenders.push(path.relative(root, file));
  }
}

if (offenders.length) {
  throw new Error(`Possible mojibake remains:\n${offenders.join('\n')}`);
}

console.log(`Checked ${files.length} miniapp files.`);
