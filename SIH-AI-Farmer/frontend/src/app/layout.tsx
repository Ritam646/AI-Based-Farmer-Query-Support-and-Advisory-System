// import type { Metadata } from "next";
// import { Inter } from "next/font/google";
// import "./globals.css"; // Ensure this import is present
// import ClientWrapper from "@/components/ClientWrapper";

// const inter = Inter({ subsets: ["latin"] });

// export const metadata: Metadata = {
//   title: "AI Farmer Assistant",
//   description: "Smart farming assistant for disease detection, market prices, and agricultural advice",
// };

// export default function RootLayout({
//   children,
// }: Readonly<{
//   children: React.ReactNode;
// }>) {
//   return (
//     <html lang="en">
//       <body className={inter.className}>
//         <ClientWrapper>{children}</ClientWrapper>
//       </body>
//     </html>
//   );
// }

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ClientWrapper from "@/components/ClientWrapper";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Farmer Assistant",
  description: "Smart farming assistant for disease detection, market prices, and agricultural advice",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ClientWrapper>{children}</ClientWrapper>
      </body>
    </html>
  );
}