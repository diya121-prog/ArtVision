import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { UploadArea } from '../components/upload/UploadArea';

describe('UploadArea Component Suite', () => {
  it('renders upload dropzone when no image is loaded', () => {
    const handleFile = vi.fn();
    const handleClear = vi.fn();

    render(
      <UploadArea
        onFileSelected={handleFile}
        selectedFile={null}
        previewUrl={null}
        onClear={handleClear}
        isLoading={false}
      />
    );

    expect(screen.getByText(/Drag and drop an artwork/i)).toBeInTheDocument();
    expect(screen.getByText(/browse files/i)).toBeInTheDocument();
    expect(screen.getByText(/Supports JPG, PNG, WebP/i)).toBeInTheDocument();
  });

  it('renders preview card when previewUrl and selectedFile are present', () => {
    const handleFile = vi.fn();
    const handleClear = vi.fn();
    const mockFile = new File(['dummy'], 'mona_lisa.jpg', { type: 'image/jpeg' });

    render(
      <UploadArea
        onFileSelected={handleFile}
        selectedFile={mockFile}
        previewUrl="data:image/svg+xml;utf8,<svg></svg>"
        onClear={handleClear}
        isLoading={false}
      />
    );

    expect(screen.getByText('mona_lisa.jpg')).toBeInTheDocument();
    expect(screen.getByText(/Loaded & Validated/i)).toBeInTheDocument();

    const clearButton = screen.getByLabelText(/Remove image and reset analysis/i);
    fireEvent.click(clearButton);
    expect(handleClear).toHaveBeenCalledTimes(1);
  });

  it('shows error banner when invalid file type is uploaded', () => {
    const handleFile = vi.fn();
    const handleClear = vi.fn();

    const { container } = render(
      <UploadArea
        onFileSelected={handleFile}
        selectedFile={null}
        previewUrl={null}
        onClear={handleClear}
        isLoading={false}
      />
    );

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const invalidFile = new File(['text'], 'notes.txt', { type: 'text/plain' });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(screen.getByText(/Upload Validation Notice/i)).toBeInTheDocument();
    expect(screen.getByText(/Unsupported file format/i)).toBeInTheDocument();
    expect(handleFile).not.toHaveBeenCalled();
  });
});
