from sanic import Sanic
from sanic_jinja2 import SanicJinja2  # pip install sanic_jinja2
from sanic import response
from sentiment_analysis import SentimentAnalysis

app = Sanic()
jinja = SanicJinja2(app)
sa = None

# ----------------------
# Serves files from the static folder to the URL /static
app.static('/img', './img')


@app.route('/')
@jinja.template('index.html')  # decorator method is static method
async def index(request):
    return


@app.route("/analyze_review", methods=['POST'])
def analyze_review(request):
    sentiment = sa.analyze_sentiment(request.form['review'][0])
    return response.json(str(sentiment[0]))


if __name__ == "__main__":
    sa = SentimentAnalysis()
    # --- Start Sanic Web Server at port 8000
    app.run(host="0.0.0.0", port=9000)
