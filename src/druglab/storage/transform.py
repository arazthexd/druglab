from __future__ import annotations
from typing import List, Any, Optional, Union, Type, Callable
from abc import ABC, abstractmethod
import logging
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    import prince  # For MCA and FAMD implementations
    PRINCE_AVAILABLE = True
except ImportError:
    PRINCE_AVAILABLE = False

from .base import BaseStorage

logger = logging.getLogger(__name__)


class BaseFeatureTransform(ABC):
    """Abstract base class for transforming features in BaseStorage objects.
    
    This class provides a framework for transforming existing features
    in storage objects, either replacing them or adding new transformed versions.
    """
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None):
        """Initialize the feature transform.
        
        Args:
            input_feature_keys: Key(s) of the input features to transform.
            output_feature_key: Key for the transformed features.
            remove_input: If True, removes input features after transformation.
            dtype: Data type for the transformed features.
        """
        if isinstance(input_feature_keys, str):
            input_feature_keys = [input_feature_keys]
        
        self.input_feature_keys = input_feature_keys
        self.output_feature_key = output_feature_key
        self.remove_input = remove_input
        self.dtype = dtype or np.float32
        self._is_fitted = False
        self._output_feature_names: Optional[List[str]] = None
    
    @property
    def is_fitted(self) -> bool:
        """Check if the transform has been fitted."""
        return self._is_fitted
    
    @property
    def output_feature_names(self) -> List[str]:
        """Get the names of the output features."""
        if self._output_feature_names is None:
            raise ValueError("Output feature names not set. Transform may not "
                             "be fitted.")
        return self._output_feature_names
    
    def _get_input_features(self, storage: BaseStorage) -> np.ndarray:
        """Extract and concatenate input features from storage."""
        feature_arrays = []
        
        for key in self.input_feature_keys:
            features = storage.features.get_features(key)
            if features is None:
                raise ValueError(f"Feature key '{key}' not found in storage")
            feature_arrays.append(features)
        
        if len(feature_arrays) == 1:
            return feature_arrays[0]
        else:
            return np.hstack(feature_arrays)
    
    def _get_input_feature_names(self, storage: BaseStorage) -> List[str]:
        """Get the names of input features."""
        all_names = []
        
        for key in self.input_feature_keys:
            metadata = storage.features.get_metadata(key)
            if metadata and 'feature_names' in metadata:
                names = metadata['feature_names']
                all_names.extend([f"{key}_{name}" for name in names])
            else:
                # Generate default names
                features = storage.features.get_features(key)
                if features is not None:
                    all_names.extend([f"{key}_{i}" 
                                      for i in range(features.shape[1])])
        
        return all_names
    
    @abstractmethod
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """Fit the transform to the input features.
        
        Args:
            features: Input features to fit the transform on.
            **kwargs: Additional arguments (e.g., labels for supervised 
                transforms).
        """
        pass
    
    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform the input features.
        
        Args:
            features: Input features to transform.
            
        Returns:
            Transformed features.
        """
        pass
    
    def fit_transform(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Fit the transform and transform the features in one step."""
        self.fit(features, **kwargs)
        return self.transform(features)
    
    @abstractmethod
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Generate names for the output features."""
        pass
    
    def apply_transform(self, 
                        storage: BaseStorage,
                        fit: bool = True,
                        **fit_kwargs) -> BaseStorage:
        """Apply the transform to storage features.
        
        Args:
            storage: The storage object containing features to transform.
            fit: If True, fits the transform on the input features.
            **fit_kwargs: Additional arguments for fitting (e.g., labels).
            
        Returns:
            The storage object with transformed features added.
        """
        if len(storage) == 0:
            logger.warning("Storage is empty, nothing to transform.")
            return storage
        
        # Check if output features already exist
        if self.output_feature_key in storage.features:
            logger.warning(
                f"Output features '{self.output_feature_key}' already exist. "
                f"They will be overwritten."
            )
        
        # Get input features
        input_features = self._get_input_features(storage)
        input_names = self._get_input_feature_names(storage)
        
        # Fit and transform
        if fit:
            print(f"Fitting and transforming features "
                  f"using {self.__class__.__name__}")
            transformed_features = self.fit_transform(input_features, 
                                                      **fit_kwargs)
        else:
            if not self.is_fitted:
                raise ValueError("Transform must be fitted before applying "
                                 "without fit=True")
            print(f"Transforming features using {self.__class__.__name__}")
            transformed_features = self.transform(input_features)
        
        # Convert to specified dtype
        transformed_features = transformed_features.astype(self.dtype)
        
        # Generate output feature names
        self._output_feature_names = \
            self._generate_output_feature_names(input_names)
        
        # Create metadata
        metadata = {
            'transform_class': self.__class__.__name__,
            'input_feature_keys': self.input_feature_keys.copy(),
            'original_shape': input_features.shape,
            'transformed_shape': transformed_features.shape,
        }
        
        # Add transformed features to storage
        storage.features.add_features(
            key=self.output_feature_key,
            features=transformed_features,
            dtype=self.dtype,
            featurizer=None,  # This is a transform, not a featurizer
            metadata=metadata
        )
        
        # Remove input features if requested
        if self.remove_input:
            for key in self.input_feature_keys:
                storage.features.remove_features(key)
                logger.info(f"Removed input features '{key}'")
        
        return storage
    
    def __call__(self, storage: BaseStorage, **kwargs) -> BaseStorage:
        """Make the transform callable."""
        return self.apply_transform(storage, **kwargs)
    
class CustomTransform(BaseFeatureTransform):
    """Custom transform using a user-provided function."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 transform_func: Callable[[np.ndarray], np.ndarray],
                 fit_func: Optional[Callable[[np.ndarray], Any]] = None,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None,
                 output_names: Optional[List[str]] = None):
        """Initialize custom transform.
        
        Args:
            input_feature_keys: Input feature keys.
            output_feature_key: Output feature key.
            transform_func: Function to transform features.
            fit_func: Optional function to fit parameters.
            remove_input: Whether to remove input features.
            dtype: Output data type.
            output_names: Names for output features.
        """
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.transform_func = transform_func
        self.fit_func = fit_func
        self.custom_output_names = output_names
        self.fit_result = None
    
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """Fit custom transform if fit function is provided."""
        if self.fit_func is not None:
            self.fit_result = self.fit_func(features, **kwargs)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using custom function."""
        if self.fit_func is not None and not self.is_fitted:
            raise ValueError("Custom transform must be fitted "
                             "before transforming")
        return self.transform_func(features)
    
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Generate output feature names."""
        if self.custom_output_names is not None:
            return self.custom_output_names.copy()
        else:
            # Generate default names based on transformation
            return [f"custom_{i}" for i in range(len(input_names))]


class CompositeTransform(BaseFeatureTransform):
    """Composite transform that applies multiple transforms in sequence."""
    
    def __init__(self, 
                 transforms: List[BaseFeatureTransform],
                 output_feature_key: str,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None):
        """Initialize composite transform.
        
        Args:
            transforms: List of transforms to apply in sequence.
            output_feature_key: Final output feature key.
            remove_input: Whether to remove initial input features.
            dtype: Output data type.
        """
        # Use input keys from first transform
        input_keys = transforms[0].input_feature_keys
        super().__init__(input_keys, output_feature_key, remove_input, dtype)
        self.transforms = transforms
    
class ScalerTransform(BaseFeatureTransform):
    """Feature scaling transform (StandardScaler, MinMaxScaler, etc.)."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 scaler_type: str = 'standard',
                 remove_input: bool = True,
                 dtype: Optional[Type[np.dtype]] = None,
                 **scaler_kwargs):
        """Initialize scaler transform.
        
        Args:
            input_feature_keys: Input feature keys.
            output_feature_key: Output feature key.
            scaler_type: Type of scaler ('standard', 'minmax', 'robust').
            remove_input: Whether to remove input features.
            dtype: Output data type.
            **scaler_kwargs: Additional arguments for the scaler.
        """
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.scaler_type = scaler_type
        self.scaler_kwargs = scaler_kwargs
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler(**scaler_kwargs)
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler(**scaler_kwargs)
        elif scaler_type == 'robust':
            self.scaler = RobustScaler(**scaler_kwargs)
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """Fit scaler to the features."""
        self.scaler.fit(features)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        return self.scaler.transform(features)
    
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Keep the same feature names for scaling."""
        return [f"scaled_{name}" for name in input_names]

class PCATransform(BaseFeatureTransform):
    """Principal Component Analysis transform."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 n_components: Optional[int] = None,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None,
                 **pca_kwargs):
        """Initialize PCA transform.
        
        Args:
            input_feature_keys: Input feature keys.
            output_feature_key: Output feature key.
            n_components: Number of components to keep.
            remove_input: Whether to remove input features.
            dtype: Output data type.
            **pca_kwargs: Additional arguments for PCA.
        """
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.n_components = n_components
        self.pca_kwargs = pca_kwargs
        self.pca = None
    
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """Fit PCA to the features."""
        self.pca = PCA(n_components=self.n_components, **self.pca_kwargs)
        self.pca.fit(features)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted PCA."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transforming")
        return self.pca.transform(features)
    
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Generate PCA component names."""
        n_components = self.pca.n_components_
        return [f"PC{i+1}" for i in range(n_components)]
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio of PCA components."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first")
        return self.pca.explained_variance_ratio_
    
class MCATransform(BaseFeatureTransform):
    """Multiple Correspondence Analysis transform for categorical data."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 n_components: Optional[int] = None,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None,
                 **mca_kwargs):
        """Initialize MCA transform.
        
        Args:
            input_feature_keys: Input feature keys (should contain 
                categorical data).
            output_feature_key: Output feature key.
            n_components: Number of components to keep.
            remove_input: Whether to remove input features.
            dtype: Output data type.
            **mca_kwargs: Additional arguments for MCA.
        """
        if not PRINCE_AVAILABLE:
            raise ImportError("prince library is required for MCA. "
                              "Install with: pip install prince")
        
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.n_components = n_components
        self.mca_kwargs = mca_kwargs
        self.mca = None
    
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """Fit MCA to the categorical features."""
        # Convert to DataFrame for MCA
        df = pd.DataFrame(features)
        
        # Ensure categorical data
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                continue
            else:
                # Convert numeric to categorical if needed
                df[col] = df[col].astype('category')
        
        self.mca = prince.MCA(n_components=self.n_components, 
                              **self.mca_kwargs)
        self.mca.fit(df)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted MCA."""
        if not self.is_fitted:
            raise ValueError("MCA must be fitted before transforming")
        
        df = pd.DataFrame(features)
        
        # Ensure categorical data
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                continue
            else:
                df[col] = df[col].astype('category')
        
        transformed = self.mca.transform(df)
        return transformed.values
    
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Generate MCA component names."""
        n_components = self.n_components or self.mca.n_components
        return [f"MCA{i+1}" for i in range(n_components)]
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio of MCA components."""
        if not self.is_fitted:
            raise ValueError("MCA must be fitted first")
        return self.mca.eigenvalues_ / self.mca.eigenvalues_.sum()
    
class FAMDTransform(BaseFeatureTransform):
    """Factor Analysis of Mixed Data transform for mixed data."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 categorical_columns: Optional[List[Union[int, str]]] = None,
                 n_components: Optional[int] = None,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None,
                 **famd_kwargs):
        """Initialize FAMD transform.
        
        Args:
            input_feature_keys: Input feature keys.
            output_feature_key: Output feature key.
            categorical_columns: Indices or names of categorical columns.
            n_components: Number of components to keep.
            remove_input: Whether to remove input features.
            dtype: Output data type.
            **famd_kwargs: Additional arguments for FAMD.
        """
        if not PRINCE_AVAILABLE:
            raise ImportError("prince library is required for FAMD. "
                              "Install with: pip install prince")
        
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.categorical_columns = categorical_columns or []
        self.n_components = n_components
        self.famd_kwargs = famd_kwargs
        self.famd = None
    
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """Fit FAMD to the mixed features."""
        df = pd.DataFrame(features)
        
        # Set categorical columns
        if self.categorical_columns:
            for col in self.categorical_columns:
                if isinstance(col, int):
                    df.iloc[:, col] = df.iloc[:, col].astype('category')
                else:
                    df[col] = df[col].astype('category')
        
        self.famd = prince.FAMD(n_components=self.n_components, 
                                **self.famd_kwargs)
        self.famd.fit(df)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted FAMD."""
        if not self.is_fitted:
            raise ValueError("FAMD must be fitted before transforming")
        
        df = pd.DataFrame(features)
        
        # Set categorical columns
        if self.categorical_columns:
            for col in self.categorical_columns:
                if isinstance(col, int):
                    df.iloc[:, col] = df.iloc[:, col].astype('category')
                else:
                    df[col] = df[col].astype('category')
        
        transformed = self.famd.transform(df)
        return transformed.values
    
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Generate FAMD component names."""
        n_components = self.famd.n_components_
        return [f"FAMD{i+1}" for i in range(n_components)]
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio of FAMD components."""
        if not self.is_fitted:
            raise ValueError("FAMD must be fitted first")
        return self.famd.eigenvalues_ / self.famd.eigenvalues_.sum()
    
class LDATransform(BaseFeatureTransform):
    """Linear Discriminant Analysis transform."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 n_components: Optional[int] = None,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None,
                 **lda_kwargs):
        """Initialize LDA transform.
        
        Args:
            input_feature_keys: Input feature keys.
            output_feature_key: Output feature key.
            n_components: Number of components to keep.
            remove_input: Whether to remove input features.
            dtype: Output data type.
            **lda_kwargs: Additional arguments for LDA.
        """
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.n_components = n_components
        self.lda_kwargs = lda_kwargs
        self.lda = None
    
    def fit(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> None:
        """Fit LDA to the features and labels."""
        self.lda = LinearDiscriminantAnalysis(n_components=self.n_components, 
                                              **self.lda_kwargs)
        self.lda.fit(features, labels)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted LDA."""
        if not self.is_fitted:
            raise ValueError("LDA must be fitted before transforming")
        return self.lda.transform(features)
    
    def _generate_output_feature_names(self, input_names: List[str]) -> List[str]:
        """Generate LDA component names."""
        n_components = self.lda.n_components \
            if hasattr(self.lda, 'n_components') else self.n_components
        return [f"LD{i+1}" for i in range(n_components)]
    
class TSNETransform(BaseFeatureTransform):
    """t-SNE transform for dimensionality reduction."""
    
    def __init__(self, 
                 input_feature_keys: Union[str, List[str]],
                 output_feature_key: str,
                 n_components: int = 2,
                 remove_input: bool = False,
                 dtype: Optional[Type[np.dtype]] = None,
                 **tsne_kwargs):
        """Initialize t-SNE transform.
        
        Args:
            input_feature_keys: Input feature keys.
            output_feature_key: Output feature key.
            n_components: Number of dimensions for t-SNE.
            remove_input: Whether to remove input features.
            dtype: Output data type.
            **tsne_kwargs: Additional arguments for t-SNE.
        """
        super().__init__(input_feature_keys, output_feature_key, 
                         remove_input, dtype)
        self.n_components = n_components
        self.tsne_kwargs = tsne_kwargs
        self.tsne = None
    
    def fit(self, features: np.ndarray, **kwargs) -> None:
        """t-SNE doesn't need separate fitting."""
        self.tsne = TSNE(n_components=self.n_components, **self.tsne_kwargs)
        self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using t-SNE."""
        if not self.is_fitted:
            self.fit(features)
        return self.tsne.fit_transform(features)
    
    def fit_transform(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Fit and transform in one step for t-SNE."""
        self.fit(features)
        return self.transform(features)
    
    def _generate_output_feature_names(self, 
                                       input_names: List[str]) -> List[str]:
        """Generate t-SNE component names."""
        return [f"tsne_{i+1}" for i in range(self.n_components)]