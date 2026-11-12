import { Suspense } from "react";

import { PlayerView } from "@/components/PlayerView";

export default function PlayerPage() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading player…</p>}>
      <PlayerView />
    </Suspense>
  );
}
