import { Suspense } from "react";

import { AccuracyView } from "@/components/AccuracyView";

export default function AccuracyPage() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading accuracy…</p>}>
      <AccuracyView />
    </Suspense>
  );
}
