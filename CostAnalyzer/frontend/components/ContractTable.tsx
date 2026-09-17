"use client";

import React from 'react';

interface ContractRow {
  contract: string;
  product_name: string;
  product_price: number;
  product_ed_izm: string;
  product_sum: number;
  Quartile: string;
  supplier_name: string;
  customer_name: string;
  OKPD2_code: string;
  OKPD2_name: string;
}

interface ContractTableProps {
  data: ContractRow[];
}

export default function ContractTable({ data }: ContractTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-slate-500">
        No results found for the selected parameters.
      </div>
    );
  }

  const getQuartileColor = (q: string) => {
    switch (q) {
      case 'Q1': return 'bg-green-100 text-green-800 border-green-200';
      case 'Q2': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'Q3': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Q4': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-slate-100 text-slate-800 border-slate-200';
    }
  };

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 shadow-sm">
      <table className="w-full text-sm text-left text-slate-600">
        <thead className="text-xs text-slate-700 uppercase bg-slate-50 border-b border-slate-200">
          <tr>
            <th className="px-4 py-3 font-semibold">Contract</th>
            <th className="px-4 py-3 font-semibold">Product</th>
            <th className="px-4 py-3 font-semibold">Price</th>
            <th className="px-4 py-3 font-semibold">Unit</th>
            <th className="px-4 py-3 font-semibold">Total</th>
            <th className="px-4 py-3 font-semibold">Quartile</th>
            <th className="px-4 py-3 font-semibold">Supplier</th>
            <th className="px-4 py-3 font-semibold">Customer</th>
            <th className="px-4 py-3 font-semibold">OKPD2</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white">
          {data.map((row, idx) => (
            <tr key={idx} className="hover:bg-slate-50 transition-colors">
              <td className="px-4 py-3 font-mono text-xs">{row.contract}</td>
              <td className="px-4 py-3 max-w-xs truncate" title={row.product_name}>
                {row.product_name}
              </td>
              <td className="px-4 py-3 font-medium">{row.product_price?.toLocaleString()}</td>
              <td className="px-4 py-3">{row.product_ed_izm}</td>
              <td className="px-4 py-3 font-medium">{row.product_sum?.toLocaleString()}</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-1 rounded-full text-[10px] font-bold border ${getQuartileColor(row.Quartile)}`}>
                  {row.Quartile}
                </span>
              </td>
              <td className="px-4 py-3 max-w-xs truncate" title={row.supplier_name}>
                {row.supplier_name}
              </td>
              <td className="px-4 py-3 max-w-xs truncate" title={row.customer_name}>
                {row.customer_name}
              </td>
              <td className="px-4 py-3 text-xs">
                <span className="font-semibold">{row.OKPD2_code}</span><br/>
                <span className="text-slate-400">{row.OKPD2_name}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
