import { Suspense } from "react";

import { PerformanceView } from "@/components/PerformanceView";

export default function PerformancePage() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading predictions…</p>}>
      <PerformanceView />
    </Suspense>
  );
}
