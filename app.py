import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest

# Load trained models
model = joblib.load('final_iris_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
selector = joblib.load('feature_selector.pkl')

# Iris class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict iris flower type
    """
    try:
        # Create feature engineering (same as training)
        data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }
        
        df = pd.DataFrame([data])
        
        # Add engineered features
        df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
        df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
        df['petal_sepal_ratio'] = df['petal_area'] / df['sepal_area']
        df['total_area'] = df['petal_area'] + df['sepal_area']
        df['length_ratio'] = df['petal length (cm)'] / df['sepal length (cm)']
        df['width_ratio'] = df['petal width (cm)'] / df['sepal width (cm)']
        
        # Feature selection (same as training)
        X_selected = selector.transform(df.values)
        
        # Scaling (same as training)
        X_scaled = scaler.transform(X_selected)
        
        # Prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Results
        predicted_class = class_names[prediction]
        confidence = probabilities[prediction] * 100
        
        # Probability breakdown
        prob_text = ""
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            prob_text += f"{class_name}: {prob*100:.1f}%\n"
        
        # Additional info about the prediction
        feature_info = (f"\nCalculated Features:\n"
                       f"• Petal Area: {df['petal_area'].iloc[0]:.2f} cm²\n"
                       f"• Sepal Area: {df['sepal_area'].iloc[0]:.2f} cm²\n"
                       f"• Petal/Sepal Ratio: {df['petal_sepal_ratio'].iloc[0]:.2f}")
        
        return (f"Prediction: {predicted_class}\n"
                f"Confidence: {confidence:.1f}%\n\n"
                f"All Probabilities:\n{prob_text}"
                f"{feature_info}")
        
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(title="Iris Flower Classifier", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # Iris Flower Classifier
    
    **Predict iris flower type using AI!**
    
    Enter your flower measurements and discover which type it is:
    - **Setosa** - Small and compact leaves, usually 4-6 cm
    - **Versicolor** - Medium balanced leaves, usually 5-7 cm  
    - **Virginica** - Large and long leaves, usually 6-8 cm
    """)
    
    # Educational section
    with gr.Accordion("About Iris Flowers", open=False):
        gr.Markdown("""
        ### Flower Anatomy:
        
        **Sepal (Outer Leaf):**
        - The outermost green leaves of the flower
        - Protects the flower bud
        - Usually 4 pieces present
        
        **Petal (Inner Leaf):**
        - The colorful, showy leaves of the flower
        - Colored to attract insects
        - Usually 3 pieces present (in iris)
        
        ### How to Measure:
        - **Length:** The longest part of the leaf (from bottom to top)
        - **Width:** The widest part of the leaf (side to side)
        - **Unit:** In centimeters (cm)
        """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Enter Flower Measurements")
            
            sepal_length = gr.Slider(
                minimum=4.0, maximum=8.0, value=5.5, step=0.1,
                label="Sepal Length (cm)", 
                info="Length of the outer leaf - the outermost green leaves"
            )
            
            sepal_width = gr.Slider(
                minimum=2.0, maximum=5.0, value=3.5, step=0.1,
                label="Sepal Width (cm)",
                info="Width of the outer leaf - measured at the widest part"
            )
            
            petal_length = gr.Slider(
                minimum=1.0, maximum=7.0, value=4.0, step=0.1,
                label="Petal Length (cm)",
                info="Length of the inner leaf - colorful, showy leaves"
            )
            
            petal_width = gr.Slider(
                minimum=0.1, maximum=3.0, value=1.5, step=0.1,
                label="Petal Width (cm)", 
                info="Width of the inner leaf - usually the narrowest part"
            )
            
            predict_btn = gr.Button("Predict!", variant="primary", size="lg")
            
            # Quick measurement guide
            gr.Markdown("""
            **Quick Measurement Guide:**
            - Use a ruler or measuring tape
            - Measure the longest/widest parts  
            - Try to measure with 0.1 cm precision
            """)
        
        with gr.Column():
            gr.Markdown("### Prediction Result")
            output = gr.Textbox(
                label="Result",
                lines=12,
                placeholder="Enter measurements and click 'Predict!'...\n\nThe system will show you:\n• Flower type\n• Confidence score\n• All probabilities\n• Calculated features"
            )
    
    # Examples with explanations
    gr.Markdown("### Example Measurements (From Real Data)")
    gr.Examples(
        examples=[
            [5.1, 3.5, 1.4, 0.2],
            [6.0, 2.7, 5.1, 1.6],
            [7.2, 3.0, 5.8, 1.6]
        ],
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        label="Try these real examples:",
        examples_per_page=3
    )
    
    # Species comparison chart
    with gr.Accordion("Species Comparison Table", open=False):
        gr.Markdown("""
        | Feature | Setosa | Versicolor | Virginica |
        |---------|------------|----------------|---------------|
        | **Sepal Length** | 4.3-5.8 cm | 4.9-7.0 cm | 4.9-7.9 cm |
        | **Sepal Width** | 2.3-4.4 cm | 2.0-3.4 cm | 2.2-3.8 cm |
        | **Petal Length** | 1.0-1.9 cm | 3.0-5.1 cm | 4.5-6.9 cm |
        | **Petal Width** | 0.1-0.6 cm | 1.0-1.8 cm | 1.4-2.5 cm |
        | **Characteristic** | Smallest | Medium size | Largest |
        | **Distinction** | Very small petals | Balanced proportions | Very long petals |
        """)
    
    gr.Markdown("""
    ---
    **Model Technical Information:**
    - **Algorithm:** Support Vector Machine (Linear Kernel)
    - **Accuracy:** 97.78% (On test data)
    - **Cross-Validation:** 98.10% (5-fold)
    - **Features:** 6 features (4 original + 2 engineered)
    - **Training Data:** 150 iris samples (50 of each type)
    - **Feature Engineering:** Petal/Sepal areas and ratios automatically calculated
    """)
    
    # Event handler
    predict_btn.click(
        fn=predict_iris,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 