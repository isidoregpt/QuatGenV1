/**
 * MoleculeImage component - Displays 2D structure of a molecule
 */

import React, { useState, useEffect } from 'react';
import { moleculeApi, RenderOptions } from '../../services/api';

interface MoleculeImageProps {
  // Either provide smiles or moleculeId
  smiles?: string;
  moleculeId?: number;

  // Display options
  width?: number;
  height?: number;
  format?: 'png' | 'svg';
  highlightQuat?: boolean;
  highlightSmarts?: string;

  // Styling
  className?: string;
  alt?: string;
  showPlaceholder?: boolean;
  placeholderText?: string;

  // Events
  onClick?: () => void;
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

export const MoleculeImage: React.FC<MoleculeImageProps> = ({
  smiles,
  moleculeId,
  width = 300,
  height = 200,
  format = 'png',
  highlightQuat = true,
  highlightSmarts,
  className = '',
  alt = 'Molecule structure',
  showPlaceholder = true,
  placeholderText = 'Loading structure...',
  onClick,
  onLoad,
  onError
}) => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadImage = async () => {
      if (!smiles && !moleculeId) {
        setError('No molecule specified');
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      const options: RenderOptions = {
        width,
        height,
        format,
        highlightQuat,
        highlightSmarts
      };

      try {
        let uri: string;

        if (moleculeId) {
          uri = await moleculeApi.getMoleculeImage(moleculeId, options);
        } else if (smiles) {
          const response = await moleculeApi.renderSmiles(smiles, options);
          uri = response.image_data_uri;
        } else {
          throw new Error('No molecule data');
        }

        setImageUri(uri);
        onLoad?.();
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load image';
        setError(errorMessage);
        onError?.(err instanceof Error ? err : new Error(errorMessage));
      } finally {
        setLoading(false);
      }
    };

    loadImage();
  }, [smiles, moleculeId, width, height, format, highlightQuat, highlightSmarts]);

  // Loading state
  if (loading && showPlaceholder) {
    return (
      <div
        className={`molecule-image-placeholder ${className}`}
        style={{ width, height }}
      >
        <div className="loading-spinner" />
        <span>{placeholderText}</span>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div
        className={`molecule-image-error ${className}`}
        style={{ width, height }}
      >
        <span className="error-icon">⚠️</span>
        <span className="error-text">{error}</span>
        {smiles && (
          <code className="smiles-fallback">{smiles.substring(0, 30)}...</code>
        )}
      </div>
    );
  }

  // Success state
  if (imageUri) {
    return (
      <img
        src={imageUri}
        alt={alt}
        width={width}
        height={height}
        className={`molecule-image ${className}`}
        onClick={onClick}
        style={{ cursor: onClick ? 'pointer' : 'default' }}
      />
    );
  }

  return null;
};

export default MoleculeImage;
