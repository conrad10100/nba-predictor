import { Suspense } from "react";

import { ModelView } from "@/components/ModelView";

export default function ModelPage() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading model…</p>}>
      <ModelView />
    </Suspense>
  );
}
