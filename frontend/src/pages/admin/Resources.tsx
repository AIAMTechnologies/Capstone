import React, { useState, useEffect, useCallback } from 'react';
import { getResources, getResource, createResource, updateResource, deleteResource } from '../../services/api';
import type { ResourcePage } from '../../types/types';

interface TreeNode {
  page: ResourcePage;
  children: TreeNode[];
}

function buildTree(pages: ResourcePage[]): TreeNode[] {
  const map = new Map<number, TreeNode>();
  const roots: TreeNode[] = [];
  const sorted = [...pages].sort((a, b) => a.sort_order - b.sort_order);
  sorted.forEach(p => map.set(p.id, { page: p, children: [] }));
  sorted.forEach(p => {
    const node = map.get(p.id)!;
    if (p.parent_id && map.has(p.parent_id)) {
      map.get(p.parent_id)!.children.push(node);
    } else {
      roots.push(node);
    }
  });
  return roots;
}

function slugify(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 8, padding: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' };
const inputStyle: React.CSSProperties = { padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14, width: '100%', boxSizing: 'border-box' as const };
const btnPrimary: React.CSSProperties = { padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' };
const btnSecondary: React.CSSProperties = { padding: '4px 10px', background: 'none', border: '1px solid #ddd', borderRadius: 4, fontSize: 12, cursor: 'pointer', color: '#666' };

const Resources: React.FC = () => {
  const [pages, setPages] = useState<ResourcePage[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [editTitle, setEditTitle] = useState('');
  const [editSlug, setEditSlug] = useState('');
  const [editContent, setEditContent] = useState('');
  const [editPublished, setEditPublished] = useState(true);
  const [isNew, setIsNew] = useState(false);
  const [newParentId, setNewParentId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);

  const fetchPages = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getResources();
      setPages(res.pages);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load resources');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchPages(); }, [fetchPages]);

  const handleSelectPage = async (id: number) => {
    setSelectedId(id);
    setIsNew(false);
    setSuccess('');
    setError('');
    try {
      const page = await getResource(id);
      setEditTitle(page.title);
      setEditSlug(page.slug);
      setEditContent(page.content || '');
      setEditPublished(page.is_published);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load page');
    }
  };

  const handleAddPage = (parentId: number | null = null) => {
    setIsNew(true);
    setSelectedId(null);
    setNewParentId(parentId);
    setEditTitle('');
    setEditSlug('');
    setEditContent('');
    setEditPublished(true);
    setSuccess('');
    setError('');
  };

  const handleTitleChange = (val: string) => {
    setEditTitle(val);
    if (isNew) setEditSlug(slugify(val));
  };

  const handleSave = async () => {
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      if (isNew) {
        const res = await createResource({
          title: editTitle,
          slug: editSlug,
          content: editContent,
          parent_id: newParentId || undefined,
        });
        setIsNew(false);
        setSelectedId(res.id);
        setSuccess('Page created');
      } else if (selectedId) {
        await updateResource(selectedId, {
          title: editTitle,
          slug: editSlug,
          content: editContent,
          is_published: editPublished,
        });
        setSuccess('Page saved');
      }
      await fetchPages();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Are you sure you want to delete this page? This cannot be undone.')) return;
    try {
      await deleteResource(id);
      if (selectedId === id) {
        setSelectedId(null);
        setEditTitle('');
        setEditSlug('');
        setEditContent('');
      }
      await fetchPages();
      setSuccess('Page deleted');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete');
    }
  };

  const tree = buildTree(pages);

  const renderTreeNode = (node: TreeNode, depth = 0): React.ReactNode => (
    <div key={node.page.id}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '8px 12px', paddingLeft: 12 + depth * 20,
          background: selectedId === node.page.id ? '#fef2f2' : 'transparent',
          borderLeft: selectedId === node.page.id ? '3px solid #c91414' : '3px solid transparent',
          cursor: 'pointer',
          fontSize: 13,
        }}
      >
        <span onClick={() => handleSelectPage(node.page.id)} style={{ flex: 1, fontWeight: selectedId === node.page.id ? 600 : 400 }}>
          {node.page.title}
          {!node.page.is_published && <span style={{ fontSize: 11, color: '#999', marginLeft: 6 }}>(draft)</span>}
        </span>
        <button onClick={() => handleAddPage(node.page.id)} style={btnSecondary} title="Add child">+</button>
        <button onClick={() => handleDelete(node.page.id)} style={{ ...btnSecondary, color: '#c91414' }} title="Delete">x</button>
      </div>
      {node.children.map(c => renderTreeNode(c, depth + 1))}
    </div>
  );

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Resources</h1>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 14 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 14 }}>{success}</div>}

      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>
        <div style={{ ...cardStyle, width: '30%', minWidth: 240 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Pages</h3>
          {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading...</p>}
          {tree.map(n => renderTreeNode(n))}
          {tree.length === 0 && !loading && <p style={{ color: '#999', fontSize: 13 }}>No pages yet</p>}
          <button onClick={() => handleAddPage(null)} style={{ ...btnPrimary, marginTop: 16, width: '100%', fontSize: 13 }}>
            Add Page
          </button>
        </div>

        <div style={{ ...cardStyle, flex: 1 }}>
          {(selectedId || isNew) ? (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>
                {isNew ? 'New Page' : 'Edit Page'}
              </h3>

              <div style={{ marginBottom: 12 }}>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Title</label>
                <input value={editTitle} onChange={e => handleTitleChange(e.target.value)} style={inputStyle} placeholder="Page title" />
              </div>

              <div style={{ marginBottom: 12 }}>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Slug</label>
                <input value={editSlug} onChange={e => setEditSlug(e.target.value)} style={inputStyle} placeholder="page-slug" />
              </div>

              <div style={{ marginBottom: 12 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 600, color: '#555', cursor: 'pointer' }}>
                  <input type="checkbox" checked={editPublished} onChange={e => setEditPublished(e.target.checked)} />
                  Published
                </label>
              </div>

              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Content (HTML)</label>
                <textarea
                  value={editContent}
                  onChange={e => setEditContent(e.target.value)}
                  style={{ ...inputStyle, height: 300, fontFamily: 'monospace', fontSize: 13, resize: 'vertical' }}
                  placeholder="<h2>Page content...</h2>"
                />
              </div>

              <button onClick={handleSave} disabled={saving} style={{ ...btnPrimary, opacity: saving ? 0.6 : 1 }}>
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          ) : (
            <p style={{ color: '#999', fontSize: 14 }}>Select a page from the tree or create a new one.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Resources;
