import { redirect } from "next/navigation";

// The Model page was renamed to Picks; keep old links working.
export default function ModelPage() {
  redirect("/picks");
}
