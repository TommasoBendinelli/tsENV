import { redirect } from 'next/navigation';
import { buildRunShortcutRedirectPath } from '../runShortcut';

export default function RunShortcutPage({ params }: { params: { run: string } }) {
  redirect(buildRunShortcutRedirectPath(params.run));
}
