import pytesseract
from PIL import Image
import numpy as np
import re
import sympy as sp
from typing import Tuple, Dict, Any
import cv2
import io

class EquationProcessor:
    def __init__(self):
        # Common physics symbols and their LaTeX representations
        self.symbol_mapping = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'pi': 'π',
            'sigma': 'σ', 'omega': 'ω', 'phi': 'φ', 'psi': 'ψ',
            'nabla': '∇', 'partial': '∂', 'infinity': '∞',
            'hbar': 'ℏ', 'dot': '·', 'ddot': '··'
        }
        
        # Common physics operators
        self.operators = {
            'derivative': ['d/dx', '∂/∂x', '∇'],
            'integral': ['∫', '∮', '∬', '∭'],
            'sum': ['∑', '∏'],
            'vector': ['→']
        }
    
    def process_equation_image(self, image_data: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Process an equation from an image.
        Returns the equation in LaTeX format and metadata.
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        # Extract equation using OCR
        equation_text = pytesseract.image_to_string(
            processed_image,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789+-*/()=[]{}.,;:<>^_\\"\'"'
        )
        
        # Convert to LaTeX format
        latex_equation = self._convert_to_latex(equation_text)
        
        # Extract metadata
        metadata = self._extract_equation_metadata(latex_equation)
        
        return latex_equation, metadata
    
    def process_equation_text(self, equation_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process an equation from text input.
        Returns the equation in LaTeX format and metadata.
        """
        # Convert to LaTeX format
        latex_equation = self._convert_to_latex(equation_text)
        
        # Extract metadata
        metadata = self._extract_equation_metadata(latex_equation)
        
        return latex_equation, metadata
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised
    
    def _convert_to_latex(self, equation_text: str) -> str:
        """
        Convert equation text to LaTeX format.
        """
        # Replace common physics symbols
        for symbol, latex in self.symbol_mapping.items():
            equation_text = equation_text.replace(symbol, latex)
        
        # Handle fractions
        equation_text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', equation_text)
        
        # Handle exponents
        equation_text = re.sub(r'(\w+)\^(\w+)', r'\1^{\2}', equation_text)
        
        # Handle subscripts
        equation_text = re.sub(r'(\w+)_(\w+)', r'\1_{\2}', equation_text)
        
        # Handle square roots
        equation_text = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', equation_text)
        
        return equation_text
    
    def _extract_equation_metadata(self, latex_equation: str) -> Dict[str, Any]:
        """
        Extract metadata from the equation.
        """
        metadata = {
            'type': self._determine_equation_type(latex_equation),
            'variables': self._extract_variables(latex_equation),
            'operators': self._extract_operators(latex_equation),
            'complexity': self._calculate_complexity(latex_equation)
        }
        return metadata
    
    def _determine_equation_type(self, equation: str) -> str:
        """
        Determine the type of equation (e.g., differential, integral, algebraic).
        """
        if any(op in equation for op in self.operators['derivative']):
            return 'differential'
        elif any(op in equation for op in self.operators['integral']):
            return 'integral'
        elif any(op in equation for op in self.operators['sum']):
            return 'summation'
        else:
            return 'algebraic'
    
    def _extract_variables(self, equation: str) -> list:
        """
        Extract variables from the equation.
        """
        # Use sympy to parse the equation
        try:
            expr = sp.sympify(equation)
            return list(expr.free_symbols)
        except:
            # Fallback to regex if sympy fails
            return re.findall(r'[a-zA-Z][a-zA-Z0-9]*', equation)
    
    def _extract_operators(self, equation: str) -> list:
        """
        Extract operators from the equation.
        """
        operators = []
        for op_type, op_list in self.operators.items():
            for op in op_list:
                if op in equation:
                    operators.append(op)
        return operators
    
    def _calculate_complexity(self, equation: str) -> int:
        """
        Calculate the complexity score of the equation.
        """
        complexity = 0
        
        # Count operators
        for op_list in self.operators.values():
            complexity += sum(equation.count(op) for op in op_list)
        
        # Count fractions
        complexity += equation.count('\\frac') * 2
        
        # Count exponents and subscripts
        complexity += equation.count('^') + equation.count('_')
        
        # Count parentheses
        complexity += equation.count('(') + equation.count(')')
        
        return complexity

# Create a singleton instance
equation_processor = EquationProcessor() 