import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Hoops Projections",
  description: "Daily NBA player points, rebounds and assists projections",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <header className="border-b border-border bg-surface">
          <nav className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-1 px-4 py-3">
            <Link href="/" className="font-semibold tracking-tight">
              Hoops Projections
            </Link>
            <Link href="/" className="text-sm text-text-2 hover:text-text">
              Slate
            </Link>
            <Link href="/accuracy" className="text-sm text-text-2 hover:text-text">
              Accuracy
            </Link>
            <Link href="/performance" className="text-sm text-text-2 hover:text-text">
              Predictions
            </Link>
            <Link href="/model" className="text-sm text-text-2 hover:text-text">
              Model
            </Link>
            <Link href="/ab-test" className="text-sm text-text-2 hover:text-text">
              A/B test
            </Link>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
