import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../lib/api';
import { Button } from '../ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Input } from '../ui/Input';
import { Upload, FileArchive, Link as LinkIcon } from 'lucide-react';

export const UploadProject = () => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [repoUrl, setRepoUrl] = useState('');
  const [creating, setCreating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.zip')) {
      setError('Please upload a ZIP file');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const project = await api.uploadProject(file);
      navigate(`/project/${project.project_id}`);
    } catch (err: any) {
      setError(err.message || 'Failed to upload project');
    } finally {
      setUploading(false);
    }
  };

  const handleCreateFromRepo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repoUrl.trim()) return;

    setCreating(true);
    setError('');

    try {
      const project = await api.createProject(repoUrl);
      navigate(`/project/${project.project_id}`);
    } catch (err: any) {
      setError(err.message || 'Failed to create project');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Upload size={20} />
            <span>Upload ZIP File</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Upload a ZIP file containing your project code
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip"
              onChange={handleFileUpload}
              className="hidden"
            />
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="w-full"
            >
              {uploading ? (
                'Uploading...'
              ) : (
                <>
                  <FileArchive size={16} className="mr-2" />
                  Choose ZIP File
                </>
              )}
            </Button>
            {error && (
              <p className="text-sm text-red-600">{error}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <LinkIcon size={20} />
            <span>Import from Git</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleCreateFromRepo} className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Import a project directly from a Git repository
            </p>
            <Input
              type="url"
              placeholder="https://github.com/username/repo"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              disabled={creating}
            />
            <Button
              type="submit"
              disabled={creating || !repoUrl.trim()}
              className="w-full"
            >
              {creating ? 'Importing...' : 'Import Repository'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};
