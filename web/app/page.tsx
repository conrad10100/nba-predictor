import { Suspense } from "react";

import { SlateView } from "@/components/SlateView";

export default function Home() {
  return (
    <Suspense fallback={<p className="text-text-2">Loading slate…</p>}>
      <SlateView />
    </Suspense>
  );
}
