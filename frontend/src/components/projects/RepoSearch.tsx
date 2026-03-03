import { useState } from 'react';
import { api } from '../../lib/api';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Spinner } from '../ui/Spinner';
import type { RepoSearchResult } from '../../types';
import { Search, FileCode } from 'lucide-react';

interface RepoSearchProps {
  projectId: string;
}

export const RepoSearch = ({ projectId }: RepoSearchProps) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RepoSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');

    try {
      const data = await api.repoSearch(projectId, query);
      // Backend returns {query: "...", results: [...]}
      const searchResults = data.results || data;
      setResults(Array.isArray(searchResults) ? searchResults : []);
    } catch (err: any) {
      setError(err.message || 'Failed to search repository');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Search size={20} />
          <span>Repository Search</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="flex space-x-2">
            <Input
              placeholder="Search code, functions, classes..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              className="flex-1"
            />
            <Button type="submit" disabled={loading}>
              {loading ? <Spinner size={16} /> : <Search size={16} />}
            </Button>
          </div>

          {error && (
            <p className="text-sm text-red-600">{error}</p>
          )}

          {results.length > 0 && (
            <div className="space-y-4 mt-6">
              <h4 className="font-medium text-sm text-foreground">Results ({results.length})</h4>
              {results.map((result, idx) => (
                <div
                  key={idx}
                  className="p-5 bg-card rounded-lg border border-border shadow-sm"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-2 flex-1">
                      <FileCode size={18} className="text-primary mt-1 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-foreground break-all">
                          {result.path || result.file_path}
                        </p>
                      </div>
                    </div>
                    {result.score && (
                      <span className="ml-2 px-2 py-1 bg-primary/10 text-primary text-xs font-medium rounded flex-shrink-0 border border-primary/20">
                        {(result.score * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  <div className="bg-muted p-4 rounded border border-border">
                    <pre className="text-sm whitespace-pre-wrap break-words overflow-x-auto">
                      <code className="text-foreground font-mono">{result.text || result.content}</code>
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          )}

          {!loading && results.length === 0 && query && (
            <p className="text-sm text-muted-foreground text-center py-4">
              No results found
            </p>
          )}
        </form>
      </CardContent>
    </Card>
  );
};
