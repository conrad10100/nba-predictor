import { redirect } from "next/navigation";

// The before and after injury pages were merged; /performance is now the after-injury view.
export default function PerformanceAfterInjuriesPage() {
  redirect("/performance");
}
