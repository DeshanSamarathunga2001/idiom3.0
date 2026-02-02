// Load idiom database locally
let idiomDatabase = [];

// Load database on page load
async function loadDatabase() {
    try {
        const response = await fetch('./data/idiom_database.json');
        idiomDatabase = await response.json();
        console.log(`✅ Loaded ${idiomDatabase.length} idioms`);
        document.getElementById('idiomCount').textContent = idiomDatabase.length;
    } catch (error) {
        console.error('❌ Error loading database:', error);
        alert('Could not load idiom database. Make sure data/idiom_database.json exists!');
    }
}

// Detect idiom in text
function detectIdiom(text) {
    const textLower = text.toLowerCase();
    for (let idiom of idiomDatabase) {
        // Remove asterisk prefix if present
        const idiomText = idiom.idiom_en.replace(/^\*\s*/, '').toLowerCase();
        if (textLower.includes(idiomText)) {
            return idiom;
        }
    }
    return null;
}

// Main function - analyze idiom
async function analyzeIdiom() {
    const inputText = document.getElementById('inputText').value.trim();
    
    if (!inputText) {
        alert('Please enter some text!');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    // Detect idiom
    const detected = detectIdiom(inputText);
    
    // Hide loading after short delay (simulate processing)
    setTimeout(() => {
        document.getElementById('loading').style.display = 'none';
        displayResults(inputText, detected);
    }, 500);
}

// Display results
function displayResults(inputText, detected) {
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Display input
    document.getElementById('outputEnglish').textContent = inputText;
    
    if (detected) {
        // Show idiom section
        document.getElementById('idiomSection').style.display = 'block';
        // Remove asterisk prefix for display
        document.getElementById('idiomText').textContent = detected.idiom_en.replace(/^\*\s*/, '');
        document.getElementById('idiomMeaning').textContent = detected.meaning || 'N/A';
        document.getElementById('idiomSinhala').textContent = detected.idiom_si;
        
        // Show example
        document.getElementById('exampleSection').style.display = 'block';
        document.getElementById('exampleEn').textContent = detected.example_en;
        document.getElementById('exampleSi').textContent = detected.example_si;
        
        // Show translation from database
        document.getElementById('outputSinhala').textContent = detected.example_si;
    } else {
        // No idiom detected
        document.getElementById('idiomSection').style.display = 'none';
        document.getElementById('exampleSection').style.display = 'none';
        document.getElementById('outputSinhala').textContent = '⚠️ No idiom detected in the input text.';
    }
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Browse random idiom
function browseRandom() {
    if (idiomDatabase.length === 0) {
        alert('Database not loaded yet!');
        return;
    }
    
    const random = idiomDatabase[Math.floor(Math.random() * idiomDatabase.length)];
    document.getElementById('inputText').value = random.example_en;
    analyzeIdiom();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDatabase();
    
    // Allow Ctrl+Enter to analyze
    document.getElementById('inputText').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeIdiom();
        }
    });
});