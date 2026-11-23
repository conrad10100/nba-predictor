import { Suspense } from "react";

import { ModelView } from "@/components/ModelView";

export default function PicksPage() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading picks…</p>}>
      <ModelView />
    </Suspense>
  );
}
