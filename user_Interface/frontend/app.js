// API endpoint
const API_URL = 'http://localhost:5000/api';

// Translate function
async function translate() {
    const inputText = document.getElementById('inputText').value.trim();
    
    if (!inputText) {
        alert('Please enter some text!');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    try {
        // Call API
        const response = await fetch(`${API_URL}/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: inputText })
        });
        
        const data = await response.json();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        // Show results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        alert('Error connecting to server. Make sure the API is running!');
    }
}

// Display results
function displayResults(data) {
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Display translation
    document.getElementById('outputEnglish').textContent = data.input;
    document.getElementById('outputSinhala').textContent = data.output;
    
    // If idiom detected, show idiom section
    if (data.idiom_detected && data.idiom_info) {
        const idiom = data.idiom_info;
        
        document.getElementById('idiomSection').style.display = 'block';
        document.getElementById('idiomText').textContent = idiom.idiom_en;
        document.getElementById('idiomMeaning').textContent = idiom.meaning || 'N/A';
        document.getElementById('idiomSinhala').textContent = idiom.idiom_si;
        
        // Show example
        document.getElementById('exampleSection').style.display = 'block';
        document.getElementById('exampleEn').textContent = idiom.example_en;
        document.getElementById('exampleSi').textContent = idiom.example_si;
    } else {
        document.getElementById('idiomSection').style.display = 'none';
        document.getElementById('exampleSection').style.display = 'none';
    }
}

// Allow Enter key to translate
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('inputText').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            translate();
        }
    });
});