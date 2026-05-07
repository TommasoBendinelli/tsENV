/** @type {import('next').NextConfig} */
const fs = require('fs');
const path = require('path');

const loadEnvFile = (filePath) => {
  if (!fs.existsSync(filePath)) return;
  const content = fs.readFileSync(filePath, 'utf8');
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const normalized = line.startsWith('export ') ? line.slice(7).trim() : line;
    const idx = normalized.indexOf('=');
    if (idx <= 0) continue;
    const key = normalized.slice(0, idx).trim();
    if (!key) continue;
    let value = normalized.slice(idx + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
};

// Load repo-root `.env` so `npm run dev` behaves like `./start.sh`.
const repoRoot = path.resolve(__dirname, '..');
loadEnvFile(path.join(repoRoot, '.env'));

if (!process.env.NEXT_PUBLIC_HUMAN_STUDY_URL) {
  const frontendHost = process.env.HUMAN_STUDY_FRONTEND_HOST || 'localhost';
  const frontendPort = process.env.HUMAN_STUDY_FRONTEND_PORT || '5173';
  const hostForUrl = frontendHost === '0.0.0.0' ? 'localhost' : frontendHost;
  process.env.NEXT_PUBLIC_HUMAN_STUDY_URL = `http://${hostForUrl}:${frontendPort}`;
}

const nextConfig = {
  reactStrictMode: false,
  webpack: (config, { isServer }) => {
    config.externals.push({
      'utf-8-validate': 'commonjs utf-8-validate',
      'bufferutil': 'commonjs bufferutil',
    });

    if (isServer && config.output) {
      config.output.chunkFilename = 'chunks/[id].js';
    }

    return config;
  },
}

module.exports = nextConfig
