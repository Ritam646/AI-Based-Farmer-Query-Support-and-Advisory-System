'use client';

import { motion } from "framer-motion";
import Link from "next/link";

interface DashboardCardProps {
  title: string;
  description: string;
  link: string;
}

export default function DashboardCard({ title, description, link }: DashboardCardProps) {
  return (
    <motion.div
      className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Link href={link}>
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300">{description}</p>
      </Link>
    </motion.div>
  );
}