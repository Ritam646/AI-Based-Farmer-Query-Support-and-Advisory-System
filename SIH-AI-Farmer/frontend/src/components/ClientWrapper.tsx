// 'use client';

// import Header from "@/components/Header";
// import ThemeToggle from "@/components/ThemeToggle";
// import Toaster from "@/components/Toaster";

// export default function ClientWrapper({
//   children,
// }: {
//   children: React.ReactNode;
// }) {
//   return (
//     <>
//       <Toaster />
//       <Header />
//       <main className="container mx-auto p-4">
//         {children}
//         <ThemeToggle />
//       </main>
//     </>
//   );
// }


'use client';

import Header from "@/components/Header";
import ThemeToggle from "@/components/ThemeToggle";
import Toaster from "@/components/Toaster";

export default function ClientWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <Toaster />
      <Header />
      <main className="container mx-auto p-4">
        {children}
        <ThemeToggle />
      </main>
    </>
  );
}