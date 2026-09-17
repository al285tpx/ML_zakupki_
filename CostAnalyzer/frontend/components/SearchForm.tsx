"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

interface SearchFormProps {
  onSearch?: (params: any) => void;
}

export default function SearchForm({ onSearch }: SearchFormProps) {
  const router = useRouter();
  const [formData, setFormData] = useState({
    product_search: '',
    product_attribute: '',
    region_code: '77',
    date_start: new Date().toISOString().split('T')[0],
    date_end: new Date().toISOString().split('T')[0],
    fz: 'All',
    price_min: '',
    price_max: '',
    okdp: '',
  });

  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          price_min: formData.price_min ? parseInt(formData.price_min) : null,
          price_max: formData.price_max ? parseInt(formData.price_max) : null,
          fz: formData.fz === 'All' ? null : formData.fz,
        }),
      });

      if (!response.ok) throw new Error('Search request failed');

      const data = await response.json();
      if (data.task_id) {
        router.push(`/results/${data.task_id}`);
      }
    } catch (error) {
      alert('Error starting search: ' + (error as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6 bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Product Search</label>
        <input
          name="product_search"
          value={formData.product_search}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
          placeholder="e.g. Бумага А4"
          required
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Attribute (Optional)</label>
        <input
          name="product_attribute"
          value={formData.product_attribute}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
          placeholder="e.g. белая"
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Region Code</label>
        <input
          name="region_code"
          value={formData.region_code}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
          required
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Date From</label>
        <input
          type="date"
          name="date_start"
          value={formData.date_start}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
          required
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Date To</label>
        <input
          type="date"
          name="date_end"
          value={formData.date_end}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
          required
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">FZ Law</label>
        <select
          name="fz"
          value={formData.fz}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
        >
          <option value="All">All</option>
          <option value="44">44-FZ</option>
          <option value="223">223-FZ</option>
        </select>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Min Price</label>
        <input
          type="number"
          name="price_min"
          value={formData.price_min}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Max Price</label>
        <input
          type="number"
          name="price_max"
          value={formData.price_max}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">OKPD2 Code</label>
        <input
          name="okdp"
          value={formData.okdp}
          onChange={handleChange}
          className="border border-slate-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
        />
      </div>

      <div className="md:col-span-2 lg:col-span-3 flex justify-end mt-4">
        <button
          type="submit"
          disabled={isLoading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-md transition-colors disabled:bg-slate-400"
        >
          {isLoading ? 'Starting Analysis...' : 'Search Contracts'}
        </button>
      </div>
    </form>
  );
}
