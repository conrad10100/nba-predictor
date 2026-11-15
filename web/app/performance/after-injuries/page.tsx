import { Suspense } from "react";

import { PerformanceView } from "@/components/PerformanceView";

export default function PerformanceAfterInjuriesPage() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading performance…</p>}>
      <PerformanceView phase="final" />
    </Suspense>
  );
}
