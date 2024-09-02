from flask import Flask, render_template, request, send_file
from textblob import TextBlob
import pandas as pd
import cleantext
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    text = request.form.get('text')
    polarity, subjectivity = None, None
    if text:
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)
    return render_template('analyze-text.html', polarity=polarity, subjectivity=subjectivity)

@app.route('/clean_text', methods=['POST'])
def clean_text():
    pre = request.form.get('pre')
    cleaned_text = None
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
    return render_template('analysis.html', cleaned_text=cleaned_text)

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    uploaded_file = request.files.get('file')
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        # Score and analyze the text in the DataFrame
        def score(x):
            blob1 = TextBlob(x)
            return blob1.sentiment.polarity

        def analyze(x):
            if x >= 0.5:
                return 'Positive'
            elif x <= -0.5:
                return 'Negative'
            else:
                return 'Neutral'

        # Apply score and analysis functions
        df['score'] = df['tweet'].apply(score)
        df['analysis'] = df['score'].apply(analyze)

        # Convert the DataFrame to HTML
        table_html = df.to_html(classes='dataframe table table-striped', index=False)

        # Convert DataFrame to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        csv_data = output.getvalue()

        # Pass the table HTML and CSV data to the template
        return render_template('analysis.html', table_html=table_html, csv_data=csv_data)

    return render_template('analysis.html')

@app.route('/download_csv', methods=['POST'])
def download_csv():
    csv_data = request.form.get('csv_data')
    if csv_data:
        return send_file(
            io.BytesIO(csv_data.encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='sentiment.csv'
        )
    return render_template('analysis.html')

if __name__ == '_main_':
    app.run(debug=True)