import Dashboard from '@/components/Dashboard';

export default function Home() {
  const runsDirName = String(process.env.WEB_MODEL_EXPLORER_RUNS_DIR_NAME ?? '').trim() || 'runs';
  return (
    <main>
      <Dashboard runsDirName={runsDirName} />
    </main>
  );
}
