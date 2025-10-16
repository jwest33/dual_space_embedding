"""Classification evaluation."""
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from loguru import logger

from embeddings.base import BaseEmbedder
from data_loaders.base import BaseDataset


class ClassificationEvaluator:
    """Evaluator for classification tasks."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        classifier: str = "logistic_regression"
    ):
        """
        Initialize classification evaluator.
        
        Args:
            embedder: Embedding model to evaluate
            classifier: Classifier to use ('logistic_regression', 'svm')
        """
        self.embedder = embedder
        self.classifier_name = classifier
        self.classifier = self._get_classifier()
        
    def _get_classifier(self):
        """Get classifier instance."""
        if self.classifier_name == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.classifier_name == "svm":
            return LinearSVC(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
    
    def evaluate(
        self,
        train_dataset: BaseDataset,
        test_dataset: BaseDataset,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate embedder on classification task.
        
        Encodes texts, trains a linear classifier, and evaluates.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(
            f"Evaluating classification on {test_dataset.name} "
            f"(train: {len(train_dataset)}, test: {len(test_dataset)})"
        )
        
        # Get train data
        train_texts = [s.text1 for s in train_dataset]
        train_labels = [s.label for s in train_dataset]
        
        # Get test data
        test_texts = [s.text1 for s in test_dataset]
        test_labels = [s.label for s in test_dataset]
        
        if None in train_labels or None in test_labels:
            raise ValueError("Dataset must have labels for classification")
        
        # Encode
        logger.debug("Encoding training texts...")
        train_embeddings = self.embedder.encode(train_texts, batch_size=batch_size)
        
        logger.debug("Encoding test texts...")
        test_embeddings = self.embedder.encode(test_texts, batch_size=batch_size)
        
        # Train classifier
        logger.debug(f"Training {self.classifier_name}...")
        self.classifier.fit(train_embeddings, train_labels)
        
        # Predict
        logger.debug("Predicting on test set...")
        predictions = self.classifier.predict(test_embeddings)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, test_labels)
        
        logger.info(f"Classification evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: List[Any]
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Dictionary with metrics
        """
        labels = np.array(labels)
        
        # Determine if binary or multi-class
        unique_labels = np.unique(labels)
        is_binary = len(unique_labels) == 2
        
        average = "binary" if is_binary else "macro"
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average=average, zero_division=0)
        recall = recall_score(labels, predictions, average=average, zero_division=0)
        f1 = f1_score(labels, predictions, average=average, zero_division=0)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "num_classes": len(unique_labels),
            "num_test_samples": len(labels),
        }
        
        return metrics
